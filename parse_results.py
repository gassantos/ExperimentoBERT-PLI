import json
import os
from pathlib import Path

def parse_gru_results(input_file, output_file):
    """
    Parse GRU results and create formatted output based on score comparison.
    
    Args:
        input_file: Path to the input JSON file with GRU results
        output_file: Path to the output JSON file to create
    """
    # Read the input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    result = {}
    
    for entry in data:
        # Skip empty entries
        if not entry or len(entry) != 2:
            continue
            
        case_pair = entry[0]
        scores = entry[1]
        
        # Skip entries without proper scores
        if not scores or len(scores) != 2:
            continue
        
        # Parse case pair (format: "case1_case2")
        parts = case_pair.split('_')
        if len(parts) != 2:
            continue
            
        case1, case2 = parts[0], parts[1]
        score1, score2 = scores[0], scores[1]
        
        # Create file names
        case1_file = f"{case1}.txt"
        case2_file = f"{case2}.txt"
        
        # Determine which case should be included based on score comparison
        if score1 > score2:
            # Case1 has higher score, so case1_file should contain case2_file
            if case1_file not in result:
                result[case1_file] = []
            result[case1_file].append(case2_file)
        else:
            # Case2 has higher score, so case2_file should contain case1_file
            if case2_file not in result:
                result[case2_file] = []
            result[case2_file].append(case1_file)
    
    # Sort the results for consistent output
    for key in result:
        result[key] = sorted(list(set(result[key])))
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Write the result to output file
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    
    print(f"Parsed {len(data)} entries and created {len(result)} case mappings")
    print(f"Output saved to: {output_file}")

def compute_metrics(labels_file, predicted_file, k_values=[1, 3, 5, 10]):
    """
    Compute precision, recall, F1-score and their @k variants.
    
    Args:
        labels_file: Path to the ground truth labels JSON file
        predicted_file: Path to the predicted results JSON file
        k_values: List of k values for @k metrics
    
    Returns:
        Dictionary containing all computed metrics
    """
    # Load data
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    with open(predicted_file, 'r') as f:
        predicted = json.load(f)
    
    # Initialize counters
    total_true_positives = 0
    total_predicted = 0
    total_actual = 0
    
    # For @k metrics
    k_metrics = {k: {'tp': 0, 'predicted': 0, 'actual': 0} for k in k_values}
    
    # Get all cases that appear in either labels or predictions
    all_cases = set(labels.keys()) | set(predicted.keys())
    
    for case in all_cases:
        true_labels = set(labels.get(case, []))
        pred_labels = predicted.get(case, [])
        
        # Convert to set for intersection
        pred_set = set(pred_labels)
        
        # Standard metrics
        tp = len(true_labels & pred_set)
        total_true_positives += tp
        total_predicted += len(pred_set)
        total_actual += len(true_labels)
        
        # @k metrics
        for k in k_values:
            pred_at_k = set(pred_labels[:k])  # Top k predictions
            tp_at_k = len(true_labels & pred_at_k)
            
            k_metrics[k]['tp'] += tp_at_k
            k_metrics[k]['predicted'] += len(pred_at_k)
            k_metrics[k]['actual'] += len(true_labels)
    
    # Calculate standard metrics
    precision = total_true_positives / total_predicted if total_predicted > 0 else 0
    recall = total_true_positives / total_actual if total_actual > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate @k metrics
    results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_cases': len(all_cases),
        'total_true_positives': total_true_positives,
        'total_predicted': total_predicted,
        'total_actual': total_actual
    }
    
    for k in k_values:
        k_tp = k_metrics[k]['tp']
        k_pred = k_metrics[k]['predicted']
        k_actual = k_metrics[k]['actual']
        
        precision_k = k_tp / k_pred if k_pred > 0 else 0
        recall_k = k_tp / k_actual if k_actual > 0 else 0
        f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0
        
        results[f'precision@{k}'] = precision_k
        results[f'recall@{k}'] = recall_k
        results[f'f1_score@{k}'] = f1_k
    
    return results

def evaluate_predictions(labels_file, predicted_file, output_file=None):
    """
    Evaluate predictions against ground truth labels and print results.
    
    Args:
        labels_file: Path to the ground truth labels JSON file
        predicted_file: Path to the predicted results JSON file
        output_file: Optional path to save results as JSON
    """
    print("Evaluating predictions...")
    print(f"Labels file: {labels_file}")
    print(f"Predicted file: {predicted_file}")
    print("-" * 50)
    
    metrics = compute_metrics(labels_file, predicted_file)
    
    # Print standard metrics
    print("Standard Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print()
    
    # Print @k metrics
    print("@k Metrics:")
    k_values = [1, 3, 5, 10]
    for k in k_values:
        if f'precision@{k}' in metrics:
            print(f"Precision@{k}: {metrics[f'precision@{k}']:.4f}")
            print(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}")
            print(f"F1-Score@{k}: {metrics[f'f1_score@{k}']:.4f}")
            print()
    
    # Print summary
    print("Summary:")
    print(f"Total cases: {metrics['total_cases']}")
    print(f"Total true positives: {metrics['total_true_positives']}")
    print(f"Total predicted: {metrics['total_predicted']}")
    print(f"Total actual: {metrics['total_actual']}")
    
    # Save results if output file is specified
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        print(f"\nResults saved to: {output_file}")
    
    return metrics

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        # Evaluation mode
        if len(sys.argv) < 4:
            print("Usage: python parse_results.py evaluate <labels.json> <predicted.json> [output.json]")
            sys.exit(1)
        
        labels_file = sys.argv[2]
        predicted_file = sys.argv[3]
        output_file = sys.argv[4] if len(sys.argv) > 4 else None
        
        evaluate_predictions(labels_file, predicted_file, output_file)
    else:
        # Original GRU parsing mode
        input_file = "output/results/gru_results.json"
        output_file = "output/results/gru_parsed_result.json"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        parse_gru_results(input_file, output_file)
