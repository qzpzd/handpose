#!/usr/bin/env python3
"""
TFLite to ONNX Converter
Converts palm_detection_lite.tflite and hand_landmark_lite.tflite to ONNX format
Ensures both models have the same input format (NCHW)
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(f"stderr: {result.stderr}")
        return False
    
    if result.stdout:
        print(result.stdout)
    return True


def convert_single_model(tflite_path, onnx_path):
    """Convert a single TFLite model to ONNX."""
    print(f"\n{'='*70}")
    print(f"Converting: {tflite_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(tflite_path):
        print(f"ERROR: TFLite file not found: {tflite_path}")
        return False
    
    # Step 1: Convert TFLite to ONNX
    if not run_command(
        f'python -m tf2onnx.convert --opset 11 --tflite {tflite_path} --output {onnx_path}',
        "Step 1: TFLite to ONNX conversion"
    ):
        return False
    
    # Step 2: Simplify ONNX model
    if not run_command(
        f'onnxsim {onnx_path} {onnx_path}',
        "Step 2: Simplify ONNX model"
    ):
        print("Warning: Simplification failed, continuing...")
    
    # Step 3: Set batch dimension
    run_command(
        f'sbi4onnx --input_onnx_file_path {onnx_path} --output_onnx_file_path {onnx_path} --initialization_character_string batch',
        "Step 3: Set batch dimension"
    )
    
    # Step 4: Rename input to 'input'
    run_command(
        f'sor4onnx --input_onnx_file_path {onnx_path} --old_new "input_1" "input" --mode inputs --output_onnx_file_path {onnx_path}',
        "Step 4: Rename input to 'input'"
    )
    
    print(f"\n✅ Conversion complete: {onnx_path}")
    return True


def analyze_onnx(onnx_path):
    """Analyze ONNX model input/output structure."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {onnx_path}")
    print(f"{'='*70}")
    
    cmd = f'python -c "import onnx; model = onnx.load(r\\"{onnx_path}\\\\"); print(\\\"Inputs:\\\\n  \\\", [inp.name + \\\": \\\" + str([d.dim_value for d in inp.type.tensor_type.shape.dim]) for inp in model.graph.input]); print(\\\"Outputs:\\\\n  \\\", [out.name + \\\": \\\" + str([d.dim_value for d in out.type.tensor_type.shape.dim]) for out in model.graph.output]); print(\\\"First node: \\\", model.graph.node[0].op_type, model.graph.node[0].name)"'
    
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Analysis failed: {result.stderr}")


def main():
    base_dir = r"D:\ai_projects\llm_sop\knowledge-assistant\palm_hand"
    
    models = [
        {
            "tflite": os.path.join(base_dir, "palm_detection_lite.tflite"),
            "onnx": os.path.join(base_dir, "palm_detection_lite.onnx")
        },
        {
            "tflite": os.path.join(base_dir, "hand_landmark_lite.tflite"),
            "onnx": os.path.join(base_dir, "hand_landmark_lite.onnx")
        }
    ]
    
    print("="*70)
    print("TFLite to ONNX Conversion")
    print("="*70)
    
    # Convert each model
    for model in models:
        if not convert_single_model(model["tflite"], model["onnx"]):
            print(f"Failed to convert {model['tflite']}")
            sys.exit(1)
        
        analyze_onnx(model["onnx"])
    
    print("\n" + "="*70)
    print("All conversions complete!")
    print("="*70)
    
    # Summary
    print("\nGenerated ONNX files:")
    for model in models:
        if os.path.exists(model["onnx"]):
            size = os.path.getsize(model["onnx"]) / 1024 / 1024
            print(f"  {model['onnx']} ({size:.2f} MB)")


if __name__ == "__main__":
    main()
