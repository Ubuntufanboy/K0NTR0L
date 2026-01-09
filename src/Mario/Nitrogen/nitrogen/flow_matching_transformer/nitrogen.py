from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.distributions import Beta
from transformers import SiglipVisionModel, AutoModel

from .modules import DiT, DiTConfig, SelfAttentionTransformer, SelfAttentionTransformerConfig

_PAD_TOKEN = 0
_IMG_TOKEN = 1
_IMG_SEP_TOKEN = 5
_LANG_TOKEN = 2
_PROPRIO_TOKEN = 3
_ACT_TOKEN = 4
_GAME_ID_TOKEN = 6


class NitroGen_Config(BaseModel):
    model_type: str = Field(default="nitrogen", frozen=True)

    add_pos_embed: bool = Field(default=False, description="Whether to add positional embedding")
    model_dtype: str = Field(default="float32", description="Model data type.")
    diffusion_model_cfg: DiTConfig = Field(..., description="Diffusion model configuration.")
    vl_self_attention_cfg: SelfAttentionTransformerConfig = Field(..., description="VL self-attention configuration.")
    hidden_size: int = Field(default=1024, description="Input embedding dimension.")
    max_seq_len: int = Field(default=1024, description="Maxium Sequence Length")
    action_dim: int  = Field(default=None, description="Action dimension.")
    action_horizon: int = Field(default=None, description="Action horizon.")
    noise_beta_alpha: float = Field(default=1.5, description="")
    noise_beta_beta: float = Field(default=1.0, description="")
    noise_s: float = Field(default=0.999, description="Flow matching noise Beta distribution s.")
    num_timestep_buckets: int = Field(default=1000, description="Number of timestep discretization buckets.")
    num_inference_timesteps: int = Field(default=None, description="Number of inference steps for noise diffusion.")
    max_num_embodiments: int = Field(default=1, description="Number of embodiments.")
    vision_encoder_name: str = Field(default="google/siglip-large-patch16-256", description="Vision encoder name.")
    vision_hidden_size: int = Field(default=768, description="Siglip hidden size.")
    add_view_embed: bool = Field(default=False, description="Whether to add view embedding.")

    tune_vision_tower: bool = Field(default=True, description="Tune vision if True.")
    tune_mm_projector: bool = Field(default=True, description="Tune mm projector if True.")
    tune_diffusion_model: bool = Field(default=True, description="Tune diffusion model if True.")
    tune_multi_projector: bool = Field(default=True, description="Tune multi projector if True.")
    tune_vl_mixing: bool = Field(default=True, description="Tune vl mixing if True.")

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "NitroGen_Config":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.model_validate(config_dict)

def swish(x):
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: shape (B, T)
        # We'll compute sin/cos frequencies across dim T
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc



class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


class NitroGen(torch.nn.Module):
    config_class = NitroGen_Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: NitroGen_Config,
        game_mapping: dict[str, int] | None = None, # Used to add a game ID token
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vision_hidden_size = config.vision_hidden_size

        if "siglip" in config.vision_encoder_name:
            model = SiglipVisionModel.from_pretrained(config.vision_encoder_name)
            self.vision_encoder = model.vision_model
            self.vision_encoder_type = "siglip"
        else:
            self.vision_encoder = AutoModel.from_pretrained(config.vision_encoder_name)
            self.vision_encoder_type = "hf_auto"
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        # self.model = instantiate(config.diffusion_model_cfg)
        self.model = DiT(config=config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        # self.vl_self_attention_model = instantiate(config.vl_self_attention_cfg)
        self.vl_self_attention_model = SelfAttentionTransformer(config=config.vl_self_attention_cfg)

        # if config.qformer_cfg is not None:
        #     self.qformer = instantiate(config.qformer_cfg)
        # else:
        #     self.qformer = nn.Identity()

        # self.state_encoder = CategorySpecificMLP(
        #     num_categories=config.max_num_embodiments,
        #     input_dim=config.max_state_dim,
        #     hidden_dim=self.hidden_size,
        #     output_dim=self.hidden_size,
        # )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.hidden_size,
            num_embodiments=config.max_num_embodiments,
        )

        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        # self.mm_vision_select_layer = config.mm_vision_select_layer
        # if config.mm_projector_cfg is not None:
        #     self.mm_projector = instantiate(config.mm_projector_cfg)
        # else:
        self.mm_projector = None
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.hidden_size)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        # if config.add_view_embed:
        #     self.view_embedding = nn.Embedding(config.max_num_views, self.hidden_size)
        #     nn.init.normal_(self.view_embedding.weight, mean=0.0, std=0.02)

        # self.vision_projector = None
        # if config.vision_hidden_size != self.hidden_size:
        #     self.vision_projector = nn.Sequential(
        #         nn.Linear(config.vision_hidden_size, self.hidden_size),
        #         nn.LayerNorm(self.hidden_size),
        #     )

        self.game_mapping = game_mapping
        # Create an embedding table for game IDs
        # Game ID tokens will be put inside vision-language tokens
        # so they need to be projected to the same dimension
        if self.game_mapping is not None:
            # 0 = unconditional
            self.game_embedding = nn.Embedding(
                len(self.game_mapping),
                self.vision_hidden_size,
                padding_idx=0,
                scale_grad_by_freq=True
            )

        self.set_trainable_parameters(
            tune_multi_projector=config.tune_multi_projector,
            tune_diffusion_model=config.tune_diffusion_model,
            tune_vision_tower=config.tune_vision_tower,
            tune_mm_projector=config.tune_mm_projector,
            tune_vl_mixing=config.tune_vl_mixing,
        )

        print(
            "total number of parameters: %e",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def set_trainable_parameters(
        self,
        tune_multi_projector: bool = True,
        tune_diffusion_model: bool = True,
        tune_vision_tower: bool = True,
        tune_mm_projector: bool = True,
        tune_vl_mixing: bool = True,
    ):
        self.tune_multi_projector = tune_multi_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vision_tower = tune_vision_tower
        self.tune_mm_projector = tune_mm_projector
        self.tune_vl_mixing = tune_vl_mixing

        for param in self.parameters():
            param.requires_grad = True
        # ### Always freeze language encoder
        # self.siglip_model.text_model.requires_grad_(False)
        # # Freeze unused parameters in siglip vision encoder
        # self.siglip_model.logit_scale.requires_grad = False
        # self.siglip_model.logit_bias.requires_grad = False

        # For siglip, we have to 
        if self.vision_encoder_type == "siglip":
            for param in self.vision_encoder.encoder.layers[11].parameters():
                param.requires_grad = False
            for param in self.vision_encoder.head.parameters():
                param.requires_grad = False


        # Freeze parameters
        if not tune_multi_projector:
            # self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.config.add_view_embed:
                self.view_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vision_tower:
            self.vision_encoder.requires_grad_(False)
        if self.mm_projector is not None and not tune_mm_projector:
            self.mm_projector.requires_grad_(False)
        if not tune_vl_mixing:
            self.vl_self_attention_model.requires_grad_(False)

        print(f"Tune action head multi_projector: {self.tune_multi_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print(f"Tune action head vision tower: {self.tune_vision_tower}")
        print(f"Tune action head mm_projector: {self.tune_mm_projector}")
        print(f"Tune action head vl_mixing: {self.tune_vl_mixing}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            # self.siglip_model.text_model.eval()
            if not self.tune_multi_projector:
                # self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
                # if self.config.add_view_embed:
                #     self.view_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vision_tower:
                self.vision_encoder.eval()
            if self.mm_projector is not None and not self.tune_mm_projector:
                self.mm_projector.eval()
            if not self.tune_vl_mixing:
                self.vl_self_attention_model.eval()

    # This function is supposedly incorrect
    # def sample_time(self, batch_size, device, dtype):
    #     sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
    #     return (self.config.noise_s - sample) / self.config.noise_s

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (1 - sample) * self.config.noise_s

    def encode_images(self, images): #, view_ids):
        batch_size, num_frames, channels, height, width = images.shape
        images = images.reshape(-1, channels, height, width)

        image_features = self.vision_encoder(images)["last_hidden_state"]
        image_features = rearrange(image_features, "(b f) n d -> b f n d", f=num_frames)

        # if self.vision_projector is not None:
        #     # change the hidden dimension of the vision features
        #     image_features = self.vision_projector(image_features)
        if self.mm_projector is not None:
            image_features = self.mm_projector(image_features)  # [B, 256, 1024] -> [B, 16, 1024]
        return image_features

    def prepare_input_embs(self, vl_token_ids, sa_token_ids, vision, action, dropped_images, game_ids=None):
        B, T = vl_token_ids.shape
        vl_embs = torch.full(
            size=(B, T, self.vision_hidden_size), fill_value=0.0, dtype=vision.dtype, device=vision.device
        )

        # Extract dimensions from vision tensor
        B, num_images, tokens_per_image, hidden_size = vision.shape

        # Create mask for _IMG_TOKEN positions
        vision_mask = (vl_token_ids == _IMG_TOKEN)  # [B, T]

        #  Flatten vision tensor over the num_images dimension
        vision_flat = vision.reshape(B, -1, self.vision_hidden_size)  # [B, T * tokens_per_image, hidden_size]

        # Create a mask for the flattened vision dimension
        # Each image contributes tokens_per_image tokens, so expand the mask accordingly
        non_dropped_mask_expanded = (dropped_images == 0).unsqueeze(-1).repeat(1, 1, tokens_per_image).reshape(B, -1)  # [B, T * tokens_per_image]

        # Select only non-dropped vision embeddings
        # This will give us the embeddings we need to place
        valid_vision_embs = vision_flat[non_dropped_mask_expanded]  # [total_valid_tokens, 1152]

        assert valid_vision_embs.shape[0] == vision_mask.sum().item(), (
            f"Number of valid vision embeddings {valid_vision_embs.shape[0]} does not match "
            f"the number of _IMG_TOKEN positions {vision_mask.sum().item()}"
        )
        # Now we need to place these at the vision_mask positions
        # Get indices where vision_mask is True
        batch_indices, token_indices = vision_mask.nonzero(as_tuple=True)

        # Place the valid embeddings at the masked positions
        vl_embs[batch_indices, token_indices] = valid_vision_embs

        # Handle Game ID tokens
        if self.game_mapping is not None and game_ids is not None:
            game_mask = vl_token_ids == _GAME_ID_TOKEN  # shape: (B, T)
            num_game_tokens = game_mask.sum().item()
            if num_game_tokens > 0:

                # Assert that each batch item has exactly one game token
                game_tokens_per_batch = game_mask.sum(dim=1)  # [B] - count of game tokens per batch item
                assert torch.all(game_tokens_per_batch == 1), (
                    f"Expected exactly 1 game token per batch item, but got: {game_tokens_per_batch.tolist()}. "
                    f"Each batch item must have exactly one _GAME_ID_TOKEN."
                )
                
                # Get game embeddings for each batch item
                game_embs = self.game_embedding(game_ids)  # [B, vision_hidden_size]
                batch_indices, token_indices = game_mask.nonzero(as_tuple=True)
                vl_embs[batch_indices, token_indices] = game_embs[batch_indices].to(dtype=vl_embs.dtype)

        # Project image separator using the learnable sep_embedding.
        sep_mask = vl_token_ids == _IMG_SEP_TOKEN  # shape: (B, T)
        num_sep = sep_mask.sum().item()
        if num_sep > 0:
            # Expand the separator embedding for each occurrence.
            repeated_sep = self.vis_sep_embedding.unsqueeze(0).expand(num_sep, self.hidden_size)
            # Assign the separator embeddings to the correct positions.
            vl_embs[sep_mask] = repeated_sep.to(dtype=vl_embs.dtype)

        B, T = sa_token_ids.shape
        sa_embs = torch.full(
            size=(B, T, self.hidden_size), fill_value=0.0, dtype=vision.dtype, device=vision.device
        )

        # Project state.
        # state_mask = sa_token_ids == _PROPRIO_TOKEN
        # state_mask = state_mask.unsqueeze(-1).expand_as(sa_embs)
        # sa_embs = sa_embs.masked_scatter(state_mask, state)

        # Project action.
        action_mask = sa_token_ids == _ACT_TOKEN
        action_mask = action_mask.unsqueeze(-1).expand_as(sa_embs)
        sa_embs = sa_embs.masked_scatter(action_mask, action)

        # Add positional embeddings
        pos_ids = torch.arange(T, dtype=torch.long, device=sa_token_ids.device)
        if self.config.add_pos_embed:
            pos_embs = self.position_embedding(pos_ids)  # (T, hidden_size)
            pos_embs = pos_embs.unsqueeze(0).expand(B, T, self.hidden_size)
            sa_embs = sa_embs + pos_embs
        return vl_embs, sa_embs

    def pack_actions(self, buttons, j_left, j_right):
        # Check that the first three dims of each input is the same
        assert buttons.shape[:3] == j_left.shape[:3] == j_right.shape[:3], (
            f"buttons shape: {buttons.shape}, "
            f"j_left shape: {j_left.shape}, "
            f"j_right shape: {j_right.shape}"
        )

        # Normalize the joysticks to 0,1
        j_left = (j_left + 1) / 2.
        j_right = (j_right + 1) / 2.

        # Concatenate the buttons and joysticks along the last dimension
        action = torch.cat([j_left, j_right, buttons], dim=-1)

        # Squeeze the second dimension of each input: this is the number of chunks, which is 1 here
        action = action.squeeze(1)
        return action

    # def unpack_actions(self, actions):
    #     # Unpack the actions into j_left, j_right, buttons
    #     j_left = actions[:, :, :2]
    #     j_right = actions[:, :, 2:4]
    #     buttons = actions[:, :, 4:]

    #     # Denormalize the joysticks back to -1,1
    #     j_left = j_left * 2. - 1.
    #     j_right = j_right * 2. - 1.

    #     # Clip into [-1,1]
    #     j_left = torch.clamp(j_left, -1, 1)
    #     j_right = torch.clamp(j_right, -1, 1)

    #     # Threshold the buttons to 0,1
    #     buttons = (buttons > 0.5).float()
    #     return j_left, j_right, buttons

    # ========= ActionHead required ============
    def forward(self, data: dict) -> dict:
        self.set_frozen_modules_to_eval_mode()

        # data = action_input
        embodiment_id = data["embodiment_id"]

        # # Check which data is present.
        # has_real_action = action_input.has_real_action
        has_real_action = data["has_real_action"]

        # 1) Encode images/text/state
        visual_features = self.encode_images(data["images"]) #, data["view_ids"])
        # text_features = self.siglip_model.text_model(
        #     input_ids=data["lang_input_ids"]
        # ).last_hidden_state
        # state_features = self.state_encoder(data["state"], embodiment_id)

        # 2) Prepare noisy trajectory
        actions = data["actions"]
        noise = torch.randn_like(actions)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # 3) Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

        # 4) Get action encoder embeddings with correct time argument
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # 5) Prepare full input to DiT (or your model)
        vl_embs, sa_embs = self.prepare_input_embs(
            data["vl_token_ids"],
            data["sa_token_ids"],
            visual_features,
            # text_features,
            # state_features,
            action_features,
            data["dropped_images"],
            game_ids=data.get("game_id"),
        )

        vl_embs = self.vl_self_attention_model(vl_embs)
        # vl_embs = self.qformer(vl_embs)
        model_output, all_hidden_states = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=data["vl_attn_mask"],
            timestep=t_discretized,
            return_all_hidden_states=True,
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # 6) Flow-matching or velocity-prediction MSE
        #    Mask for variable-length trajectories
        mask = data["actions_mask"]  # shape => (B, seq_len_of_actions, ...)
        raw_loss = F.mse_loss(pred_actions, velocity, reduction="none")
        mask = has_real_action[:, None, None] * mask
        raw_loss = raw_loss * mask
        action_loss = (has_real_action[:, None, None] * raw_loss).sum() / (mask.sum() + 1e-6)

        loss = action_loss

        return {
            "loss": loss,
        }

    @torch.inference_mode()
    def get_action(self, data: dict, old_layout:bool = False) -> dict:
        """
        For i in [0..N-1]:
          1) t = i/N
          2) velocity = model(x(t), t)
          3) x(t + dt) = x(t) + dt * velocity
        """

        # data = action_input
        embodiment_id = data["embodiment_id"]

        batch_size = data["images"].shape[0]
        device = data["images"].device
        dtype = data["images"].dtype
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=dtype,
            device=device,
        )

        # 1) Hyperparameters for flow sampling
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # 2) Encode static context (images, text, state) once if it does not depend on actions
        visual_features = self.encode_images(data["images"]) #, data["view_ids"])
        # text_features = self.siglip_model.text_model(
        #     input_ids=data["lang_input_ids"]
        # ).last_hidden_state
        # state_features = self.state_encoder(data["state"], embodiment_id)

        # 3) Start denoising the actions
        for i in range(num_steps):
            # ---- (a) Discretize continuous time in [0,1]
            t_cont = i / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # ---- (b) Build embeddings (actions included)
            # Pass the *current* actions at time t into the action encoder
            action_features = self.action_encoder(
                actions,
                (torch.ones(actions.shape[0]) * t_discretized).to(device),
                embodiment_id,
            )
            vl_embs, sa_embs = self.prepare_input_embs(
                data["vl_token_ids"],
                data["sa_token_ids"],
                visual_features,
                # text_features,
                # state_features,
                action_features,
                data["dropped_images"],
                game_ids=data["game_ids"],
            )
            vl_embs = self.vl_self_attention_model(vl_embs)
            # vl_embs = self.qformer(vl_embs)
            # ---- (c) Forward pass to get velocity = d/dt x(t)
            timesteps = torch.from_numpy(np.array([t_discretized])).to(device).long()
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=data["vl_attn_mask"],
                timestep=timesteps,
            )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -actions.shape[1] :]

            # ---- (d) Naive Euler step: x(t + dt) = x(t) + dt * velocity
            actions = actions + dt * pred_velocity

        return {
            "action_tensor": actions,
        }

    @torch.inference_mode()
    def get_action_with_cfg(self, data_cond: dict, data_uncond: dict, cfg_scale: float = 1.0) -> dict:
        """
        Use a form of classifier free guidance to sample actions. This can only be used on
        models that were trained on multiple frames of actions. The idea is that we sample
        velocity with and without the frame history, and then we push the sampled actions
        towards the ones that were sampled with the frame history.

        data_with_hist = conditional input
        data_without_hist = unconditional input
        
        This function works with any kind of conditioning, not just history.

        For i in [0..N-1]:
          1) t = i/N
          2) velocity = (1 - cfg_scale) * model(x(t), t, None) + cfg_scale * model(x(t), t, history)
          3) x(t + dt) = x(t) + dt * velocity
        """

        # data = action_input
        embodiment_id = data_cond["embodiment_id"]

        batch_size = data_cond["images"].shape[0]
        device = data_cond["images"].device
        dtype = data_cond["images"].dtype
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=dtype,
            device=device,
        )

        # 1) Hyperparameters for flow sampling
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # 2) Encode static context (images, text, state) once if it does not depend on actions
        visual_features_cond = self.encode_images(data_cond["images"])
        visual_features_uncond = self.encode_images(data_uncond["images"])
        # text_features = self.siglip_model.text_model(
        #     input_ids=data["lang_input_ids"]
        # ).last_hidden_state
        # state_features = self.state_encoder(data["state"], embodiment_id)

        # 3) Start denoising the actions
        for i in range(num_steps):
            # ---- (a) Discretize continuous time in [0,1]
            t_cont = i / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # ---- (b) Build embeddings (actions included)
            # Pass the *current* actions at time t into the action encoder
            action_features = self.action_encoder(
                actions,
                (torch.ones(actions.shape[0]) * t_discretized).to(device),
                embodiment_id,
            )

            # Predict velocity with history
            vl_embs, sa_embs = self.prepare_input_embs(
                data_cond["vl_token_ids"],
                data_cond["sa_token_ids"],
                visual_features_cond,
                action_features,
                data_cond["dropped_images"],
            )
            vl_embs = self.vl_self_attention_model(vl_embs)
            # ---- (c) Forward pass to get velocity = d/dt x(t)
            timesteps = torch.from_numpy(np.array([t_discretized])).to(device).long()
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=data_cond["vl_attn_mask"],
                timestep=timesteps,
            )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity_cond = pred[:, -actions.shape[1] :]

            # Predict velocity without history
            vl_embs, sa_embs = self.prepare_input_embs(
                data_uncond["vl_token_ids"],
                data_uncond["sa_token_ids"],
                visual_features_uncond,
                action_features,
                data_uncond["dropped_images"],
            )
            vl_embs = self.vl_self_attention_model(vl_embs)
            # ---- (c) Forward pass to get velocity = d/dt x(t)
            timesteps = torch.from_numpy(np.array([t_discretized])).to(device).long()
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=data_uncond["vl_attn_mask"],
                timestep=timesteps,
            )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity_uncond = pred[:, -actions.shape[1] :]

            # ---- (d) Combine velocities with cfg_scale
            pred_velocity = pred_velocity_cond + cfg_scale * (pred_velocity_cond - pred_velocity_uncond)

            # ---- (e) Naive Euler step: x(t + dt) = x(t) + dt * velocity
            actions = actions + dt * pred_velocity

        return {
            "action_tensor": actions,
        }

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
