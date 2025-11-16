"""
Inference script for interactive prompt policy (Scheme A).

Placeholder for future implementation.

This would run inference with a trained prompt policy,
visualizing the interactive prompting process.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Inference for prompt policy (Scheme A)")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="prompt_infer")
    args = parser.parse_args()
    
    print("Scheme A (Interactive Prompt Policy) inference is not yet implemented.")
    print("This is a placeholder for future work.")
    print("")
    print("Expected functionality:")
    print("- Load trained prompt policy")
    print("- Run multi-step prompting on test images")
    print("- Visualize click positions and mask evolution")
    print("- Save JSON stats (IoU, num_clicks, etc.)")


if __name__ == "__main__":
    main()

