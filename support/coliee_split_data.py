import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_jsonl_file(input_file, train_file, valid_file, train_ratio=0.7):
    """
    Split a JSONL file into train and valid sets while maintaining label balance.
    
    Args:
        input_file: Path to input JSONL file
        train_file: Path to output train JSONL file
        valid_file: Path to output valid JSONL file
        train_ratio: Proportion for training data (default 0.7)
    """
    # Read all data from JSONL file
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Separate by label
    label_0 = [item for item in data if item['label'] == 0]
    label_1 = [item for item in data if item['label'] == 1]
    
    print(f"Total samples: {len(data)}")
    print(f"Label 0: {len(label_0)}, Label 1: {len(label_1)}")
    
    # Split each label group maintaining proportions
    train_0, valid_0 = train_test_split(label_0, train_size=train_ratio, random_state=42)
    train_1, valid_1 = train_test_split(label_1, train_size=train_ratio, random_state=42)
    
    # Combine splits
    train_data = train_0 + train_1
    valid_data = valid_0 + valid_1
    
    # Write train file
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    # Write valid file
    with open(valid_file, 'w') as f:
        for item in valid_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\nTrain set: {len(train_data)} samples")
    print(f"Valid set: {len(valid_data)} samples")
    print(f"Train - Label 0: {len(train_0)}, Label 1: {len(train_1)}")
    print(f"Valid - Label 0: {len(valid_0)}, Label 1: {len(valid_1)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split JSONL file into train and valid sets with label balance")
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("train_file", help="Path to output train JSONL file")
    parser.add_argument("valid_file", help="Path to output valid JSONL file")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Proportion for training data (default 0.7)")
    
    args = parser.parse_args()
    
    split_jsonl_file(args.input_file, args.train_file, args.valid_file, args.train_ratio)
