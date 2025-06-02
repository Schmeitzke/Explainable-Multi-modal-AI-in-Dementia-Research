import argparse
import os
from train import train

def main():
    parser = argparse.ArgumentParser(description="Train AD-Transformer on ADNI data")
    
    parser.add_argument("--train_csv", type=str, 
                        default="",
                        help="Path to the training data CSV file.")
    parser.add_argument("--test_csv", type=str,
                        default="",
                        help="Path to the test data CSV file.")
    
    parser.add_argument("--config", type=str, default="ADtransformer_pytorch/config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="ADtransformer_pytorch/output/3class_results/", 
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
        
    train(args.config, args.train_csv, args.test_csv, args.output_dir)
    
    print(f"Training completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 