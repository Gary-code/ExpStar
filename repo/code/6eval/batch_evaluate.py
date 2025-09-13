#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import jsonlines
from tqdm import tqdm
from evaluate import load
# Assuming my_cider and CHAIR are available locally or in the environment
from my_cider import CHAIR
import argparse
import numpy as np
import pandas as pd
import glob
import torch
import re # Import regex module

# Load evaluation metrics
# Ensure you have run `pip install evaluate bertscore rouge meteor sacrebleu transformers accelerate`
BLEU = load("bleu")
ROUGE = load("rouge")
METEOR = load("meteor")
# Specify a suitable model for BERTScore. This one requires downloading.
# Consider using a smaller model if GPU memory is limited, e.g., 'bert-base-uncased'
BERTSCORE = load("bertscore")

def find_files(directory, postfix=""):
    """Finds files with a specific postfix in the given directory and its subdirectories."""
    output_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(postfix):
                output_files.append(os.path.join(root, file))
    return output_files

def find_files_in_first_level(directory, postfix=""):
    """Finds files with a specific postfix only in the first level of the given directory."""
    if not os.path.isdir(directory):
         print(f"Warning: Directory not found: {directory}")
         return []
    return glob.glob(os.path.join(directory, f"*{postfix}"))

def convert_jsonl_format(input_path, output_path=None):
    """
    Converts a JSONL file from 'labels' and 'response' fields to 'gt' and 'pred' format.
    If output_path is None, generates a default path in a 'convert' subdirectory.
    """
    if output_path is None:
        # Automatically generate output path
        dirname = os.path.dirname(input_path)
        basename = os.path.basename(input_path)
        output_dir = os.path.join(dirname, "converted") # Use a generic subdir name
        os.makedirs(output_dir, exist_ok=True)
        # Add a prefix to indicate it's converted
        output_path = os.path.join(output_dir, f"converted_{basename}")

    print(f"Converting file format: {input_path} -> {output_path}")

    # Perform the conversion
    try:
        with jsonlines.open(input_path, mode="r") as reader, jsonlines.open(output_path, mode="w") as writer:
            for obj in reader:
                # Convert fields to 'gt' and 'pred' structure
                new_obj = {
                    "gt": {"feedback": obj.get("labels", "")},
                    "pred": {"feedback": obj.get("response", "")}
                }
                writer.write(new_obj)
        print(f"Conversion complete, results saved to: {output_path}")
        return output_path
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        return None
    except Exception as e:
        print(f"An error occurred during conversion of {input_path}: {e}")
        return None


def process_file(file_path, data_type=None, batch_size=30):
    """
    Processes a single JSON/JSONL file:
    - Extracts references (refs) and predictions (preds).
    - Optionally filters data by data_type.
    - Performs evaluation using loaded metrics.
    - Returns a dictionary of evaluation results.
    """
    refs = []
    preds = []

    # Read the file
    data = []
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif file_path.endswith('.jsonl'):
            with jsonlines.open(file_path, mode='r') as reader:
                for obj in reader:
                    data.append(obj)
        else:
            print(f"Error: Unsupported file format for {file_path}. Must be .json or .jsonl")
            return "error, unsupported file format"
    except FileNotFoundError:
        print(f"Error: Input file not found: {file_path}")
        return "error, file not found"
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON from {file_path}")
         return "error, invalid JSON"
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        return f"error, reading file failed: {e}"


    # Filter data based on data_type and collect refs/preds
    filtered_data = []
    for entry in data:
        if data_type is None or entry.get("data_type", "") == data_type:
             # Ensure 'gt' and 'pred' fields exist and contain 'feedback'
             if "gt" in entry and "feedback" in entry["gt"] and "pred" in entry and "feedback" in entry["pred"]:
                 filtered_data.append(entry)
             else:
                 # print(f"Warning: Skipping entry missing 'gt.feedback' or 'pred.feedback' in {file_path}")
                 pass # Silently skip entries with missing required fields for evaluation


    if not filtered_data:
         print(f"No data found for evaluation (after filtering by data_type='{data_type}' if applicable).")
         return "error, no data to evaluate"

    for entry in filtered_data:
        refs.append([entry["gt"]["feedback"]]) # References should be a list of lists
        preds.append(entry["pred"]["feedback"])


    if refs:
        results = evaluate_metrics(preds, refs, batch_size)
        return results
    else:
        return "error, no valid data found after filtering"


def evaluate_metrics(preds, refs, batch_size=30):
    """
    Calculates evaluation metrics and returns a results dictionary.
    Uses batching for BERTScore to manage GPU memory.
    """
    print(f"Evaluating {len(preds)} samples...")

    # Prepare data for metrics - CHAIR expects a specific format
    imgids = [str(i) for i in range(len(preds))] # Use string IDs
    # CHAIR requires reference dictionary and prediction dictionary
    # Format: {imgid: [ref1, ref2, ...]} for references, {imgid: pred} for predictions
    ref_dict = {imgid: ref_list for imgid, ref_list in zip(imgids, refs)}
    pred_dict = {imgid: pred for imgid, pred in zip(imgids, preds)}

    # Ensure CHAIR is initialized correctly. The second argument is typically a path, often not strictly needed for compute_metric if data is passed directly.
    # Assuming CHAIR can compute from dictionaries. Adjust initialization if necessary based on my_cider implementation.
    # For now, assuming a placeholder path is acceptable if needed.
    evaluator = CHAIR(ref_dict, "") # Pass ref_dict directly

    try:
        # CHAIR computation - Adjust method call based on actual my_cider implementation
        # The provided CHAIR class has compute_metric with (imgids, preds, refs)
        # Let's stick to that signature if it works, but the standard is dicts.
        # If compute_metric expects lists, imgids might not be used internally for lookup.
        # Let's try the list version first as in the original code.
        # Ensure refs format is list of lists: [[ref1], [ref2], ...]
        cider = evaluator.compute_metric(imgids, preds, refs)
        # If compute_metric fails or expects dicts, try:
        # cider = evaluator.compute_metric(pred_dict, ref_dict) # This is the more standard way
    except Exception as e:
        print(f"Error calculating CIDer: {e}")
        cider = float('nan') # Indicate failure


    # Compute other metrics
    try:
        bleu_score = BLEU.compute(predictions=preds, references=refs)
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        bleu_score = {"precisions": [float('nan')]*4, "bleu": float('nan')}

    try:
        rouge = ROUGE.compute(predictions=preds, references=refs)
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")
        rouge = {'rougeL': float('nan'), 'rouge1': float('nan'), 'rouge2': float('nan')} # Include other common ROUGE scores

    try:
        meteor = METEOR.compute(predictions=preds, references=refs)
    except Exception as e:
        print(f"Error calculating METEOR: {e}")
        meteor = {'meteor': float('nan')}


    # BERTScore computation with batching
    bertscore_precision_mean = float('nan')
    bertscore_recall_mean = float('nan')
    bertscore_f1_mean = float('nan')
    try:
        # Check if a GPU is available and move model there for speed
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device} for BERTScore")
        # The BERTSCORE load function usually handles device automatically if transformers is installed.
        # Pass device=device might be needed depending on the evaluate library version and setup.
        # Let's rely on the default behavior first, which usually uses GPU if available.
        # If OOM errors occur, reducing batch_size is the primary solution.

        # Note: BERTScore compute function automatically handles batching if batch_size is provided.
        bertscore = BERTSCORE.compute(
            predictions=preds,
            references=refs,
            lang="en", # Specify language
            model_type="microsoft/deberta-large-mnli", # Or other suitable model
            batch_size=batch_size # Use the specified batch size
            # device=device # Add this if device auto-detection is not reliable
        )

        # Ensure scores are not empty before calculating mean
        if bertscore and "precision" in bertscore and bertscore["precision"]:
             bertscore_precision_mean = float(np.mean(bertscore["precision"]))
        if bertscore and "recall" in bertscore and bertscore["recall"]:
             bertscore_recall_mean = float(np.mean(bertscore["recall"]))
        if bertscore and "f1" in bertscore and bertscore["f1"]:
             bertscore_f1_mean = float(np.mean(bertscore["f1"]))

    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        # Specific handling for CUDA OOM
        if "CUDA out of memory" in str(e):
            print("CUDA Out of Memory error during BERTScore calculation. Try reducing the --batch_size.")
        pass # Keep BERTScore values as NaN


    # Collect results in the specified order
    results = {
        "BLEU-1": bleu_score["precisions"][0],
        "BLEU-2": bleu_score["precisions"][1],
        "BLEU-3": bleu_score["precisions"][2],
        "BLEU-4": bleu_score["precisions"][3],
        "METEOR": meteor.get('meteor', float('nan')), # Use .get() with default for safety
        "ROUGE-L-F": rouge.get('rougeL', float('nan')), # Use .get() with default for safety
        "CIDer": cider,
        "BERTScore": {
            "Precision": bertscore_precision_mean,
            "Recall": bertscore_recall_mean,
            "F1": bertscore_f1_mean,
        }
    }
    return results

def save_to_csv(results, experiment_name, output_csv_dir, data_type=None):
    """
    Saves evaluation results to a CSV file.
    Numeric results are formatted as percentages with two decimal places.
    """
    # Use the provided output directory for the CSV
    results_dir = output_csv_dir

    # Ensure the directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Use a fixed name for the CSV file within the specified directory
    csv_path = os.path.join(results_dir, 'evaluation_results.csv') # Generic CSV name

    # Convert float values to percentage strings with two decimal places
    def format_percentage(value):
        if pd.isna(value): # Handle NaN values
            return ''
        return f"{value * 100:.2f}"

    # Prepare the new row of results
    new_row = {
        'experiment': experiment_name,
        'data_type': data_type if data_type else 'all', # 'all' if no data_type filter applied
        'BLEU-1': format_percentage(results.get('BLEU-1')),
        'BLEU-2': format_percentage(results.get('BLEU-2')),
        'BLEU-3': format_percentage(results.get('BLEU-3')),
        'BLEU-4': format_percentage(results.get('BLEU-4')),
        'METEOR': format_percentage(results.get('METEOR')),
        'ROUGE-L-F': format_percentage(results.get('ROUGE-L-F')),
        'CIDer': format_percentage(results.get('CIDer')),
        'BERTScore-p': format_percentage(results.get('BERTScore', {}).get('Precision')),
        'BERTScore-r': format_percentage(results.get('BERTScore', {}).get('Recall')),
        'BERTScore-f1': format_percentage(results.get('BERTScore', {}).get('F1')),
    }

    # Read the existing CSV file (if it exists)
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # Define columns explicitly for a new DataFrame
        columns = ['experiment', 'data_type', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4',
                   'METEOR', 'ROUGE-L-F', 'CIDer', 'BERTScore-p', 'BERTScore-r', 'BERTScore-f1']
        df = pd.DataFrame(columns=columns)
    except pd.errors.EmptyDataError:
         # Handle case where file exists but is empty
         columns = ['experiment', 'data_type', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4',
                    'METEOR', 'ROUGE-L-F', 'CIDer', 'BERTScore-p', 'BERTScore-r', 'BERTScore-f1']
         df = pd.DataFrame(columns=columns)
    except Exception as e:
         print(f"An error occurred while reading CSV file {csv_path}: {e}")
         # Create an empty DataFrame as fallback
         columns = ['experiment', 'data_type', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4',
                    'METEOR', 'ROUGE-L-F', 'CIDer', 'BERTScore-p', 'BERTScore-r', 'BERTScore-f1']
         df = pd.DataFrame(columns=columns)


    # Append the new result row to the DataFrame
    new_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_df], ignore_index=True)

    # Save to CSV file
    try:
        df.to_csv(csv_path, index=False)
        print(f"\nResults appended to: {csv_path}")
    except IOError as e:
        print(f"Error writing to CSV file {csv_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing CSV file {csv_path}: {e}")


def needs_conversion(file_path):
    """
    Checks if a JSONL file needs format conversion from 'labels'/'response' to 'gt'/'pred'.
    Reads only the first line to determine format. Returns False for non-JSONL files.
    """
    if not file_path.lower().endswith('.jsonl'):
        return False

    try:
        with jsonlines.open(file_path, mode="r") as reader:
            # Try to read the first object
            try:
                obj = next(reader)
                # Check for the presence of original fields
                if "labels" in obj and "response" in obj:
                    # Also check for the absence of the new fields to be sure it's the old format
                    if "gt" not in obj and "pred" not in obj:
                        return True
            except StopIteration:
                 # File is empty, no conversion needed/possible
                 return False
            except json.JSONDecodeError:
                 print(f"Warning: Could not decode first line of {file_path}. Assuming no conversion needed.")
                 return False # Cannot determine format if first line is invalid
    except FileNotFoundError:
        print(f"Warning: File not found when checking conversion need: {file_path}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while checking conversion needs for {file_path}: {e}")
        return False # Assume no conversion needed on error


def process_directory(directory_path, output_csv_dir, convert=False, batch_size=15):
    """
    Processes all first-level jsonl files within a directory.
    Optionally converts file format and performs evaluation, saving results to CSV.
    """
    print(f"Processing directory: {directory_path}")

    # Find all first-level jsonl files
    jsonl_files = find_files_in_first_level(directory_path, postfix=".jsonl")

    if not jsonl_files:
        print(f"No first-level .jsonl files found in directory: {directory_path}")
        return

    print(f"Found {len(jsonl_files)} jsonl files:")
    for i, file in enumerate(jsonl_files, 1):
        print(f"{i}. {os.path.basename(file)}")

    # Process each file
    for file_path in jsonl_files:
        print(f"\n--- Processing file: {file_path} ---")

        # Determine if conversion is needed or forced
        needs_conv = convert or needs_conversion(file_path)

        if needs_conv:
            print("Conversion of file format detected/forced.")
            # Output path for conversion is handled by convert_jsonl_format function (creates subdir)
            json_path = convert_jsonl_format(file_path)
            if not json_path: # Skip if conversion failed
                print(f"Skipping evaluation due to failed conversion for {file_path}")
                continue
        else:
            json_path = file_path
            print("File format is already suitable or conversion is not required.")

        # Extract experiment name from the processed file path
        experiment_name = os.path.basename(json_path).replace('.jsonl', '').replace('.json', '')
        print(f"Experiment name: {experiment_name}")

        # Evaluate all data in the file
        print("Starting evaluation for all data...")
        results_all = process_file(json_path, batch_size=batch_size) # Pass batch_size

        if isinstance(results_all, dict):
            save_to_csv(results_all, experiment_name, output_csv_dir) # Pass output_csv_dir
            print(f"Evaluation results for all data: {results_all}")
        else:
            print(f"Evaluation failed for all data: {results_all}")

        # Get all unique data_types if they exist in the dataset
        data_types = set()
        data = [] # Reload data if needed, or pass data from process_file if possible
        try:
             if json_path.endswith('.json'):
                 with open(json_path, 'r', encoding='utf-8') as f:
                     data = json.load(f)
             elif json_path.endswith('.jsonl'):
                 with jsonlines.open(json_path, mode='r') as reader:
                     for obj in reader:
                         data.append(obj)
        except Exception as e:
             print(f"Error reloading data from {json_path} to find data_types: {e}")
             data = [] # Ensure data is empty on error

        for entry in data:
            if "data_type" in entry:
                data_types.add(entry["data_type"])

        # Evaluate for each data_type separately
        if data_types:
            print(f"Found specific data types: {data_types}. Evaluating each type...")
            for data_type in sorted(list(data_types)): # Sort for consistent output
                print(f"\nStarting evaluation for data type: {data_type}...")
                results = process_file(json_path, data_type=data_type, batch_size=batch_size) # Pass data_type and batch_size
                if isinstance(results, dict):
                    save_to_csv(results, experiment_name, output_csv_dir, data_type) # Pass output_csv_dir and data_type
                    print(f"Evaluation results for data type '{data_type}': {results}")
                else:
                    print(f"Evaluation failed for data type '{data_type}': {results}")
        else:
            print("No specific data types found for separate evaluation.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate NLP metrics and convert JSONL file format.")
    # Made --src required and removed default path
    parser.add_argument("--src", type=str, required=True,
                        help="Input file or directory path to evaluate.")
    parser.add_argument("--convert", action="store_true",
                        help="Force conversion of file format even if not detected.")
    parser.add_argument("--output", type=str,
                        help="Output file path for the converted file (used only with --convert). If not provided, a default path in a 'converted' subdir is used.")
    parser.add_argument("--directory", action="store_true",
                        help="Treat the input source path as a directory containing multiple files to evaluate.")
    parser.add_argument("--batch_size", type=int, default=15,
                        help="Batch size for BERTScore calculation to manage GPU memory usage.")
    # Added argument for output CSV directory
    parser.add_argument("--output_csv_dir", type=str, default="./evaluation_results",
                        help="Directory to save the evaluation results CSV file.")

    args = parser.parse_args()

    print(f"Processing source: {args.src}")
    print(f"Using batch size for BERTScore: {args.batch_size}")
    print(f"Saving results CSV to: {args.output_csv_dir}")

    # Determine if the source is a directory
    is_directory = args.directory or os.path.isdir(args.src)

    if is_directory:
        # Process the entire directory
        process_directory(args.src, args.output_csv_dir, args.convert, args.batch_size) # Pass output_csv_dir and batch_size
    else:
        # Process a single file
        input_path = args.src
        print(f"Input is a file: {input_path}")

        # Check if conversion is needed or forced
        needs_conv = args.convert or needs_conversion(input_path)

        if needs_conv:
            print("Conversion of file format detected/forced.")
            output_path = args.output if args.output else None # Use provided output path if available
            json_path = convert_jsonl_format(input_path, output_path)
            if not json_path: # Skip if conversion failed
                 print(f"Skipping evaluation due to failed conversion for {input_path}")
                 return
        else:
            json_path = input_path
            print("File format is already suitable or conversion is not required.")


        # Extract experiment name from the processed file path
        experiment_name = os.path.basename(json_path).replace('.json', '').replace('.jsonl', '')
        print(f"Experiment name: {experiment_name}")

        # Evaluate all data in the file
        print("Starting evaluation for all data...")
        results_all = process_file(json_path, batch_size=args.batch_size) # Pass batch_size

        if isinstance(results_all, dict):
            save_to_csv(results_all, experiment_name, args.output_csv_dir) # Pass output_csv_dir
            print(f"Evaluation results for all data: {results_all}")
        else:
            print(f"Evaluation failed for all data: {results_all}")

        # Get all unique data_types if they exist in the dataset
        data_types = set()
        data = [] # Reload data if needed, or pass data from process_file if possible
        try:
            if json_path.endswith('.json'):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif json_path.endswith('.jsonl'):
                with jsonlines.open(json_path, mode='r') as reader:
                    for obj in reader:
                        data.append(obj)
        except Exception as e:
             print(f"Error reloading data from {json_path} to find data_types: {e}")
             data = [] # Ensure data is empty on error


        for entry in data:
            if "data_type" in entry:
                data_types.add(entry["data_type"])

        # Evaluate for each data_type separately
        if data_types:
            print(f"Found specific data types: {data_types}. Evaluating each type...")
            for data_type in sorted(list(data_types)): # Sort for consistent output
                print(f"\nStarting evaluation for data type: {data_type}...")
                results = process_file(json_path, data_type=data_type, batch_size=args.batch_size) # Pass data_type and batch_size
                if isinstance(results, dict):
                    save_to_csv(results, experiment_name, args.output_csv_dir, data_type) # Pass output_csv_dir and data_type
                    print(f"Evaluation results for data type '{data_type}': {results}")
                else:
                    print(f"Evaluation failed for data type '{data_type}': {results}")
        else:
             print("No specific data types found for separate evaluation.")


if __name__ == "__main__":
    main()