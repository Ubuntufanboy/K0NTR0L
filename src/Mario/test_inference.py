
import os
import sys
import torch
import numpy as np

# Add Nitrogen to path
sys.path.append(os.path.join(os.getcwd(), "Nitrogen"))
from nitrogen.inference_session import load_model, InferenceSession

NITROGEN_CKPT = "Nitrogen/ng.pt"

def test_inference():
    print("Loading model...")
    ckpt_path = os.path.abspath(NITROGEN_CKPT)
    model, tokenizer, img_proc, ckpt_config, game_mapping, action_downsample_ratio = load_model(ckpt_path)
    
    selected_game = None
    if game_mapping:
        candidates = [g for g in game_mapping.keys() if "Mario" in g and "Bros" in g]
        if candidates:
            selected_game = candidates[0]
            print(f"Selected game: {selected_game}")
    
    session = InferenceSession(
        model,
        ckpt_path,
        tokenizer,
        img_proc,
        ckpt_config,
        game_mapping,
        selected_game,
        old_layout=False,
        cfg_scale=1.0,
        action_downsample_ratio=action_downsample_ratio,
        context_length=None
    )
    
    print("Model loaded. Preparing input...")
    
    # Create a dummy frame (256x256 RGB)
    frame_rgb = np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8)
    
    print("Running predict...")
    try:
        preds = session.predict(frame_rgb)
        print("Prediction successful!")
        print(preds)
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
