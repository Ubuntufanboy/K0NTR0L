"""
Multi-Modal Encoders: Text and Vision embedding modules
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel
import timm
from torchvision import transforms

logger = logging.getLogger(__name__)


class TextEncoder:
    """BERT-mini text encoder for command embeddings"""
    
    def __init__(self, model_name: str = "prajjwal1/bert-mini", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        
        logger.info(f"Loading text encoder: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"Text encoder loaded. Embedding dim: {self.embedding_dim}")
    
    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to embedding vector"""
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        outputs = self.model(**inputs)
        
        # Use [CLS] token embedding
        embedding = outputs.last_hidden_state[:, 0, :]
        
        return embedding.squeeze(0)  # Return 1D tensor
    
    def encode_batch(self, texts: list) -> torch.Tensor:
        """Encode batch of texts"""
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use [CLS] token embeddings
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings


class VisionEncoder:
    """MobileViTv2 vision encoder for visual state embeddings"""
    
    def __init__(self, 
                 model_name: str = "mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k",
                 device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        
        logger.info(f"Loading vision encoder: {model_name}")
        
        # Load pretrained model
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
        ).to(device)
        self.model.eval()
        
        # Get model info
        data_config = timm.data.resolve_model_data_config(self.model)
        self.input_size = data_config['input_size'][1]  # Should be 544
        
        # Create transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=data_config['mean'],
                std=data_config['std']
            )
        ])
        
        # Get embedding dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(device)
            dummy_output = self.model(dummy_input)
            self.embedding_dim = dummy_output.shape[1]
        
        logger.info(f"Vision encoder loaded. Input size: {self.input_size}, "
                   f"Embedding dim: {self.embedding_dim}")
    
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model"""
        
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Handle grayscale
        if len(frame.shape) == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        
        # Apply transform
        tensor = self.transform(frame)
        
        return tensor
    
    @torch.no_grad()
    def encode(self, frame: np.ndarray) -> torch.Tensor:
        """Encode single frame to embedding"""
        
        tensor = self.preprocess(frame).unsqueeze(0).to(self.device)
        embedding = self.model(tensor)
        
        return embedding.squeeze(0)  # Return 1D tensor
    
    @torch.no_grad()
    def encode_batch(self, frames: list) -> torch.Tensor:
        """Encode batch of frames"""
        
        tensors = torch.stack([
            self.preprocess(frame) for frame in frames
        ]).to(self.device)
        
        embeddings = self.model(tensors)
        
        return embeddings


class SensorEncoder(nn.Module):
    """Simple MLP for encoding raw sensor/kinematic data"""
    
    def __init__(self, input_dim: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_dim = embedding_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        ).to(self.device)
        
        logger.info(f"Sensor encoder loaded. Input dim: {input_dim}, "
                   f"Embedding dim: {self.embedding_dim}")
    
    def forward(self, x: np.ndarray) -> torch.Tensor:
        """Encode sensor data"""
        # Ensure input is a flat tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        x = x.to(self.device).flatten()
        return self.network(x)


class MultiModalEncoder:
    """Combined text, vision, and sensor encoder"""
    
    def __init__(self, 
                 kinematics_dim: int,
                 text_model: str = "prajjwal1/bert-mini",
                 vision_model: str = "mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k",
                 device: str = "cuda"):
        
        self.device = device
        
        logger.info("Initializing multi-modal encoder...")
        
        self.text_encoder = TextEncoder(text_model, device)
        self.vision_encoder = VisionEncoder(vision_model, device)
        self.sensor_encoder = SensorEncoder(kinematics_dim)
        
        self.text_dim = self.text_encoder.embedding_dim
        self.vision_dim = self.vision_encoder.embedding_dim
        self.sensor_dim = self.sensor_encoder.embedding_dim
        self.total_dim = self.text_dim + self.vision_dim + self.sensor_dim
        
        # Cache for current command embedding
        self.cached_command = None
        self.cached_command_embedding = None
        
        logger.info(f"Multi-modal encoder ready. "
                   f"Text dim: {self.text_dim}, "
                   f"Vision dim: {self.vision_dim}, "
                   f"Sensor dim: {self.sensor_dim}, "
                   f"Total dim: {self.total_dim}")
    
    def set_command(self, command: str):
        """Set and cache command embedding"""
        
        if command != self.cached_command:
            self.cached_command = command
            self.cached_command_embedding = self.text_encoder.encode(command)
            logger.debug(f"Cached new command embedding: {command[:50]}...")
    
    def encode_state(self, obs_dict: dict) -> torch.Tensor:
        """Encode state (uses cached command + current frame + kinematics)"""
        
        if self.cached_command_embedding is None:
            raise ValueError("Command not set. Call set_command() first.")
        
        # Extract data from observation dictionary
        frame = obs_dict.get('frame')
        kinematics = obs_dict.get('kinematics', obs_dict.get('obs'))
        
        if frame is None or kinematics is None:
            raise ValueError("Observation dictionary must contain 'frame' and 'kinematics' or 'obs'")
        
        # Encode modalities
        vision_embedding = self.vision_encoder.encode(frame)
        sensor_embedding = self.sensor_encoder(kinematics)
        
        # Concatenate embeddings
        combined = torch.cat([
            self.cached_command_embedding,
            vision_embedding,
            sensor_embedding
        ])
        
        return combined
    
    def encode_batch(self, command: str, obs_dicts: list) -> torch.Tensor:
        """Encode batch of states"""
        
        # Encode text once
        text_embedding = self.text_encoder.encode(command)
        
        # Extract and encode frames and kinematics
        frames = [d['frame'] for d in obs_dicts]
        kinematics_list = [d.get('kinematics', d.get('obs')) for d in obs_dicts]
        
        vision_embeddings = self.vision_encoder.encode_batch(frames)
        
        # ToDo: Batch sensor encoding
        sensor_embeddings = torch.stack([
            self.sensor_encoder(k) for k in kinematics_list
        ])
        
        # Expand text embedding to batch size
        batch_size = vision_embeddings.shape[0]
        text_batch = text_embedding.unsqueeze(0).expand(batch_size, -1)
        
        # Concatenate
        combined = torch.cat([text_batch, vision_embeddings, sensor_embeddings], dim=1)
        
        return combined


class EncoderWrapper(nn.Module):
    """Wrapper to use multi-modal encoder in PPO network"""
    
    def __init__(self, multimodal_encoder: MultiModalEncoder):
        super().__init__()
        self.encoder = multimodal_encoder
        self.output_dim = multimodal_encoder.total_dim
        
        # Freeze encoders
        for param in self.encoder.text_encoder.model.parameters():
            param.requires_grad = False
        for param in self.encoder.vision_encoder.model.parameters():
            param.requires_grad = False
            
        # Sensor encoder can be trainable
        for param in self.encoder.sensor_encoder.parameters():
            param.requires_grad = True
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - observations should already be encoded
        This is mainly for compatibility with standard RL frameworks
        """
        return observations
    
    def encode_observation(self, obs_dict: dict) -> torch.Tensor:
        """Encode raw observation dictionary"""
        return self.encoder.encode_state(obs_dict)

