import time
import json
from collections import deque

import torch
import numpy as np

from transformers import AutoImageProcessor
from nitrogen.flow_matching_transformer.nitrogen import NitroGen, NitroGen_Config
from nitrogen.mm_tokenizers import NitrogenTokenizerConfig, NitrogenTokenizer, Tokenizer
from nitrogen.cfg import CkptConfig
from nitrogen.shared import PATH_REPO

def summarize_parameters(module, name='model', depth=0, max_depth=3):
    """
    Print a tree-like summary of parameters in a PyTorch module.
    
    Args:
        module: PyTorch module to summarize
        name: Name of the module (for root level)
        depth: Current depth in the tree
        max_depth: Maximum depth to traverse
    """
    if depth > max_depth:
        return
    
    # Count total parameters in this module
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # Print indented summary
    indent = "  " * depth
    print(f"{indent}{name}: {total_params:,} params ({trainable_params:,} trainable)")
    
    # Recursively summarize submodules
    if depth < max_depth:
        for child_name, child_module in module.named_children():
            summarize_parameters(child_module, child_name, depth + 1, max_depth)


def load_model(checkpoint_path: str):
    """Load model and args from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = CkptConfig.model_validate(checkpoint["ckpt_config"])
    model_cfg = ckpt_config.model_cfg
    tokenizer_cfg = ckpt_config.tokenizer_cfg

    print("Checkpoint args:")
    print(json.dumps(ckpt_config.model_dump(), indent=4))

    # Initialize tokenizer and language model
    img_proc = AutoImageProcessor.from_pretrained(model_cfg.vision_encoder_name)

    # Create VLM with pre-loaded language model
    if isinstance(model_cfg, NitroGen_Config):
        assert isinstance(tokenizer_cfg, NitrogenTokenizerConfig), \
            "NitroGen_Config requires NitrogenTokenizerConfig for tokenization"
        tokenizer_cfg.training = False
        if tokenizer_cfg.game_mapping_cfg is not None:
            tokenizer_cfg.game_mapping_cfg.src_files = [
                x.replace("/mnt/amlfs-02/shared/gaming/gamingvla", str(PATH_REPO))
                for x in tokenizer_cfg.game_mapping_cfg.src_files
            ]
        tokenizer = NitrogenTokenizer(tokenizer_cfg)
        game_mapping = tokenizer.game_mapping
        model = NitroGen(config=model_cfg, game_mapping=game_mapping)
        # model.num_inference_timesteps = 16
        action_downsample_ratio = 1
    else:
        raise ValueError(f"Unsupported model config type: {type(model_cfg)}")

    summarize_parameters(model, max_depth=3)

    print(model)

    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer.eval()
    model.to("cuda")

    return model, tokenizer, img_proc, ckpt_config, game_mapping, action_downsample_ratio

class InferenceSession:
    """Manages state for a single inference session."""
    
    def __init__(
        self,
        model,
        ckpt_path: str,
        tokenizer: Tokenizer,
        img_proc,
        ckpt_config: CkptConfig,
        game_mapping: dict,
        selected_game: str,
        old_layout: bool,
        cfg_scale: float,
        action_downsample_ratio: float,
        context_length=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.img_proc = img_proc
        self.ckpt_config = ckpt_config
        self.game_mapping = game_mapping
        self.selected_game = selected_game
        self.old_layout = old_layout
        self.cfg_scale = cfg_scale
        self.action_downsample_ratio = action_downsample_ratio
        self.ckpt_path = ckpt_path

        # Load modality config
        self.modality_config = self.ckpt_config.modality_cfg

        self.max_buffer_size = context_length if context_length is not None else self.modality_config.frame_per_sample
        self.action_interleaving = self.modality_config.action_interleaving
        self.is_flowmatching = isinstance(self.ckpt_config.model_cfg, NitroGen_Config)

        # Buffers
        self.obs_buffer = deque(maxlen=self.max_buffer_size)
        self.action_buffer = deque(maxlen=self.max_buffer_size)

    @classmethod
    def from_ckpt(cls, checkpoint_path: str, old_layout=False, cfg_scale=1.0, context_length=None):
        """Create an InferenceSession from a checkpoint."""
        model, tokenizer, img_proc, ckpt_config, game_mapping, action_downsample_ratio = load_model(checkpoint_path)

        if game_mapping is not None:
            # Ask user to pick a game from the list
            print("Available games in tokenizer mapping:")
            for game, idx in game_mapping.items():
                print(f"{idx:03d}: {game}")
            selected_game = input("Enter the game ID to use (leave empty for unconditional): ")
            if selected_game == "":
                selected_game = None
            else:
                selected_idx = int(selected_game)
                assert selected_idx in game_mapping.values(), f"Invalid game ID {selected_idx}"

                candidates = [k for k,v in game_mapping.items() if v == selected_idx]
                assert len(candidates) == 1, f"Multiple games found for ID {selected_idx}: {candidates}"

                selected_game = candidates[0]
        else:
            selected_game = None
            print("No game mapping available, proceeding without game conditioning")

        return cls(
            model,
            checkpoint_path,
            tokenizer,
            img_proc,
            ckpt_config,
            game_mapping,
            selected_game,
            old_layout,
            cfg_scale,
            action_downsample_ratio,
            context_length
        )

    def info(self):
        return {
            "ckpt_path": self.ckpt_path,
            "selected_game": self.selected_game,
            "old_layout": self.old_layout,
            "cfg_scale": self.cfg_scale,
            "context_length": self.max_buffer_size,
            "action_interleaving": self.action_interleaving,
            "is_flowmatching": self.is_flowmatching,
            "action_downsample_ratio": self.action_downsample_ratio,
        }

    def reset(self):
        """Reset all buffers."""
        self.obs_buffer.clear()
        self.action_buffer.clear()

    def predict(self, obs):
        start_time = time.time()

        current_frame = self.img_proc([obs], return_tensors="pt")["pixel_values"]
        self.obs_buffer.append(current_frame)
        
        # Prepare model inputs
        pixel_values = torch.cat(list(self.obs_buffer), dim=0)
    
        if self.action_interleaving and len(self.action_buffer) > 0:
            action_tensors = {
                key: torch.cat([a[key] for a in list(self.action_buffer)], dim=0)
                for key in ["buttons", "j_left", "j_right"]
            }
        else:
            action_tensors = {"buttons": None, "j_left": None, "j_right": None}

        # Run inference
        if self.is_flowmatching:
            predicted_actions = self._predict_flowmatching(pixel_values, action_tensors)
        else:
            predicted_actions = self._predict_ar(pixel_values, action_tensors)
        
        # Add to action buffer
        self.action_buffer.append(predicted_actions)
        
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.3f}s")

        # Convert to list of action dicts
        n_actions = len(predicted_actions["buttons"])
        j_left = predicted_actions["j_left"].squeeze().cpu().numpy()
        j_right = predicted_actions["j_right"].squeeze().cpu().numpy()
        buttons = predicted_actions["buttons"].squeeze().cpu().numpy()

        return {
            "j_left": j_left,
            "j_right": j_right,
            "buttons": buttons,
        }

    def _predict_flowmatching(self, pixel_values, action_tensors):

        available_frames = len(self.obs_buffer)
        frames = torch.zeros((self.max_buffer_size, *pixel_values.shape[1:]), 
                            dtype=pixel_values.dtype, device="cuda")
        frames[-available_frames:] = pixel_values
        dropped_frames = torch.zeros((self.max_buffer_size,), dtype=torch.bool, device="cuda")
        dropped_frames[:self.max_buffer_size - available_frames] = True
        
        data_with_history = {
            "frames": frames,
            "dropped_frames": dropped_frames,
            "game": self.selected_game
        }
        tokenized_data_with_history = self.tokenizer.encode(data_with_history)
        
        frame_mask = torch.ones((self.max_buffer_size,), dtype=torch.bool, device="cuda")
        frame_mask[-1] = False
        data_without_history = {
            "frames": frames,
            "dropped_frames": frame_mask,
            "game": None
        }
        tokenized_data_without_history = self.tokenizer.encode(data_without_history)
        
        # Convert to CUDA tensors with batch dimension
        for tokenized_data in [tokenized_data_with_history, tokenized_data_without_history]:
            for k, v in tokenized_data.items():
                if isinstance(v, torch.Tensor):
                    tokenized_data[k] = v.unsqueeze(0).to("cuda")
                elif isinstance(v, np.ndarray):
                    tokenized_data[k] = torch.tensor(v, device="cuda").unsqueeze(0)
                else:
                    tokenized_data[k] = [v]
        
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if self.cfg_scale == 1.0:
                    model_output = self.model.get_action(tokenized_data_with_history, 
                                                        old_layout=self.old_layout)
                else:
                    model_output = self.model.get_action_with_cfg(
                        tokenized_data_with_history,
                        tokenized_data_without_history,
                        cfg_scale=self.cfg_scale
                    )
                predicted_actions = self.tokenizer.decode(model_output)
        
        return predicted_actions
