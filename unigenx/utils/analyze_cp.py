import argparse
import torch
import os



def analyze_checkpoint(ckpt_path):
    # load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    print("all params：")
    for param_name, _ in state_dict.items():
        print(param_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze PyTorch Checkpoint parameters")
    parser.add_argument("-p", required=True, help="Path to the checkpoint file")

    args = parser.parse_args()
    analyze_checkpoint(args.p)