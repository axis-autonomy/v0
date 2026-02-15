import torch
import os
import sys
from PIL import Image

# 1. Path Setup
PROJECT_ROOT = os.path.abspath(os.getcwd())
TEPNET_SRC = os.path.join(PROJECT_ROOT, "train-ego-path-detection", "src")
if TEPNET_SRC not in sys.path:
    sys.path.insert(0, TEPNET_SRC)

from utils.interface import Detector

def export_twinkling_to_onnx():
    model_name = "twinkling-rocket-21"
    # Adjust this path if your weights are in a different folder
    MODEL_DIR = os.path.join(PROJECT_ROOT, "train-ego-path-detection", "egopath", "weights", model_name)
    OUTPUT_ONNX = os.path.join(MODEL_DIR, f"{model_name}.onnx")
    
    if not os.path.exists(MODEL_DIR):
        print(f"❌ Error: Model directory not found at {MODEL_DIR}")
        return

    print(f"🔄 Loading {model_name} (EfficientNet-B3)...")
    
    # FIX: Added None for crop_coords to satisfy the positional argument requirement
    try:
        detector = Detector(
            model_path=MODEL_DIR, 
            crop_coords=None,      # <--- Added this fix
            runtime='pytorch', 
            device='cpu'
        )
    except Exception as e:
        print(f"❌ Failed to initialize Detector: {e}")
        return

    model = detector.model
    model.eval()

    # 2. Dummy Input
    # Using 512x512 as a standard resolution for EfficientNet-B3 in this context
    dummy_input = torch.randn(1, 3, 512, 512)

    print(f"🚀 Exporting to ONNX: {OUTPUT_ONNX}...")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            OUTPUT_ONNX,
            export_params=True,
            opset_version=18, 
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✅ Success! ONNX file created.")
    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    export_twinkling_to_onnx()