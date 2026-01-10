import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MarioDataset
from policy import PolicyNetwork
from encoders import MultiModalEncoder
import logging
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 1. Initialize Encoders
    # We use a dummy kinematics dim as we don't use sensor data
    encoders = MultiModalEncoder(kinematics_dim=1, device=device)
    
    # 2. Initialize Dataset
    # Relative path from src/mm_ppo to src/Mario/Nitrogen/out/ng
    runs_dir = os.path.join(os.path.dirname(__file__), "../Mario/Nitrogen/out/ng")
    run_ids = ["0003", "0004"]
    
    logger.info("Initializing Dataset...")
    dataset = MarioDataset(
        runs_dir=runs_dir,
        run_ids=run_ids,
        encoders=encoders,
        seq_len=16,
        target_fps=15,
        original_fps=60
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 3. Initialize Policy
    text_dim = encoders.text_dim
    vision_dim = encoders.vision_dim
    action_dim = 21 # Defined by unified controller keys
    
    logger.info("Initializing Policy...")
    policy = PolicyNetwork(
        input_dim=0, # Unused
        action_dim=action_dim,
        text_dim=text_dim,
        vision_dim=vision_dim,
        sensor_dim=0, # Unused
        continuous=True
    ).to(device)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-1)
    loss_fn = nn.MSELoss()
    
    # 4. Training Loop
    logger.info("Starting Training...")
    policy.train()
    
    num_epochs = 5
    
    for epoch in range(num_epochs):
        total_loss = 0
        steps = 0
        
        for i, (vision, text, actions) in enumerate(loader):
            vision = vision.to(device)
            text = text.to(device)
            actions = actions.to(device)
            
            # Prepare input: [Text | Vision] (Sensor is ignored/implicit)
            # Expand text to sequence length
            seq_len = vision.shape[1]
            text_seq = text.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Concatenate
            x = torch.cat([text_seq, vision], dim=-1)
            
            # Forward pass
            # Returns mean, log_std, hidden
            mean, log_std, _ = policy(x)
            
            # Calculate Loss (MSE on mean vs target actions)
            loss = loss_fn(mean, actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / max(1, steps)
        logger.info(f"Epoch {epoch} Completed. Average Loss: {avg_loss:.4f}")
        
    # Save Model
    save_path = os.path.join(os.path.dirname(__file__), "mario_imitation_policy.pt")
    torch.save(policy.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
