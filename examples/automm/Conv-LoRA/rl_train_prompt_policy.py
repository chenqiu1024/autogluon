"""
RL training for interactive prompt policy (Scheme A).

Placeholder for future implementation.

This would train a policy to select optimal points/boxes for SAM
using GRPO with reward = ΔIoU - λ·click_cost.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="RL training for prompt policy (Scheme A)")
    parser.add_argument("--task", type=str, default="polyp")
    parser.add_argument("--output_dir", type=str, default="rl_prompt")
    args = parser.parse_args()
    
    print("Scheme A (Interactive Prompt Policy) is not yet implemented.")
    print("This is a placeholder for future work.")
    print("")
    print("Expected functionality:")
    print("- Train policy to select points/boxes for SAM")
    print("- Multi-step episodes with ΔIoU rewards")
    print("- GRPO training with PERL regularizers")
    print("")
    print("See README.md section 5.6 for details.")


if __name__ == "__main__":
    main()

