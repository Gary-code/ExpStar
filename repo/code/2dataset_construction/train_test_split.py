#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import shutil
import logging
from tqdm import tqdm
from datetime import datetime
import argparse

def setup_logging(log_base_dir="./logs"):
    """Set up logging."""
    # Use a configurable base directory for logs
    log_dir = os.path.join(log_base_dir, "split_logs") # Use a subdirectory specific to splitting
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"train_test_split_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return log_file

def extract_video_key(video_path):
    """
    Extract a unique identifier for a video from its path.
    This identifier should represent the experiment/unit/subject,
    not the individual step clip.

    For example: From "/home/.../processed/biology/bio_35/video/04-Experiment Title/04-Experiment Title_step_1.mp4"
    extract "biology/bio_35/04-Experiment Title"

    Args:
        video_path: Path to the video file.
    Returns:
        str: Unique video identifier.
    """
    # Normalize and split the path
    normalized_path = os.path.normpath(video_path)
    parts = normalized_path.split(os.sep)

    # Find the position of "processed"
    try:
        # Assuming the structure is .../processed/subject/unit_id/video/experiment_name/clip_name
        # We want subject/unit_id/experiment_name
        idx = parts.index("processed")
        # Check if there are enough parts after 'processed'
        if idx + 4 < len(parts):
            subject = parts[idx + 1]
            unit = parts[idx + 2]
            # Skip "video" folder (parts[idx + 3])
            experiment = parts[idx + 4]

            # Return subject/unit/experiment as the unique identifier
            return f"{subject}{os.sep}{unit}{os.sep}{experiment}"
        else:
            # If structure is unexpected, fall back to a simpler key
             logging.warning(f"Unexpected path structure for key extraction: {video_path}. Falling back to simpler key.")
             # Attempt to get subject/unit/filename without step
             if idx + 2 < len(parts):
                 subject = parts[idx + 1]
                 unit = parts[idx + 2]
                 # Try to get the main video name part before _step_X.mp4
                 filename = os.path.basename(video_path)
                 match = re.match(r'(.*?)(_step_\d+)?\.\w+', filename)
                 if match:
                     experiment_part = match.group(1)
                     return f"{subject}{os.sep}{unit}{os.sep}{experiment_part}"
                 else:
                      return os.path.basename(video_path) # Fallback to just filename
             else:
                 return os.path.basename(video_path) # Fallback to just filename


    except (ValueError, IndexError):
        # If "processed" or subsequent parts are not found, return the filename as fallback
        logging.warning(f"'processed' not found in path or path too short: {video_path}. Falling back to filename.")
        return os.path.basename(video_path)

def split_dataset(source_dir, output_dir, train_ratio=10, test_ratio=1):
    """
    Splits the JSONL dataset into training and testing sets based on video groups.
    Ensures all steps from the same original video (experiment) stay together.

    Args:
        source_dir: Source data directory (containing subject folders).
        output_dir: Output directory for split data.
        train_ratio: Training set ratio.
        test_ratio: Testing set ratio.
    """
    # Get all subject directories, excluding 'merge' or similar processing folders
    subject_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d)) and d not in ["merge", "split", "logs", "processed"]] # Exclude potential processing/output folders

    # Create training and testing directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    # Ensure output directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Dictionary to record statistics
    stats = {
        "subjects": {},
        "total": {
            "train": 0,
            "test": 0,
            "train_groups": 0, # Count of unique video groups in train
            "test_groups": 0   # Count of unique video groups in test
        }
    }

    # Process each subject
    for subject in sorted(subject_dirs):
        logging.info(f"Processing subject: {subject}")
        subject_path = os.path.join(source_dir, subject)

        # Get all unit directories within the subject
        unit_dirs = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]

        # Dictionary to store all data items grouped by video key
        video_groups = {}

        # Collect all data items and their group key
        all_data_items = []

        # Process JSONL files in each unit
        for unit in sorted(unit_dirs):
            unit_path = os.path.join(subject_path, unit)
            jsonl_files = [f for f in os.listdir(unit_path) if f.endswith(".jsonl")]

            for jsonl_file in jsonl_files:
                jsonl_path = os.path.join(unit_path, jsonl_file)

                try:
                    with open(jsonl_path, 'r', encoding='utf-8') as f:
                        for line_idx, line in enumerate(f):
                            try:
                                data = json.loads(line)

                                # Check if data format is correct and contains video path
                                if "videos" in data and data["videos"] and isinstance(data["videos"], list) and data["videos"][0]:
                                    video_path = data["videos"][0]

                                    # Extract the video group identifier
                                    video_key = extract_video_key(video_path)

                                    if video_key not in video_groups:
                                        video_groups[video_key] = []

                                    # Record the data item along with its source info and key
                                    data_item = {
                                        "jsonl_file": jsonl_path,
                                        "line_idx": line_idx,
                                        "data": data,
                                        "video_key": video_key
                                    }

                                    video_groups[video_key].append(data_item)
                                    all_data_items.append(data_item) # Keep a flat list for easier processing later
                                else:
                                    logging.warning(f"Data item missing video information: {jsonl_path}, line {line_idx+1}")

                            except json.JSONDecodeError:
                                logging.error(f"JSON decoding error: {jsonl_path}, line {line_idx+1}")
                            except Exception as e:
                                logging.error(f"Error processing data item: {jsonl_path}, line {line_idx+1}: {str(e)}")

                except IOError as e:
                    logging.error(f"Error reading file {jsonl_path}: {e}")


        # Split video groups based on the ratio
        group_keys = list(video_groups.keys())
        if not group_keys:
            logging.warning(f"No video groups found in subject {subject}, skipping.")
            continue

        random.seed(42)  # Set random seed for reproducibility
        random.shuffle(group_keys)

        total_groups = len(group_keys)
        total_ratio = train_ratio + test_ratio
        if total_ratio <= 0:
             logging.error(f"Invalid train/test ratio sum ({total_ratio}). Please ensure train_ratio + test_ratio > 0.")
             continue # Skip this subject or exit, depending on desired behavior

        # Calculate split counts, ensuring at least one group in each set if possible
        train_group_count = max(1, int(total_groups * train_ratio / total_ratio)) if total_groups >= 2 and test_ratio > 0 else total_groups # Put all in train if test_ratio is 0
        if total_groups > 1 and test_ratio > 0:
             # Ensure test set gets at least one group if total_groups > 1 and test_ratio > 0
             if total_groups - train_group_count == 0:
                 if train_group_count > 1: # Only shrink train if it has more than one group
                     train_group_count -= 1
                 else: # If train only has one, put it in test to ensure test is not empty
                     train_group_count = 0
        elif total_groups == 1 and test_ratio > 0:
             train_group_count = 0 # Put the single group in test if test_ratio > 0

        train_keys = set(group_keys[:train_group_count])
        test_keys = set(group_keys[train_group_count:])

        logging.info(f"Subject {subject} split results: {len(train_keys)} groups for train, {len(test_keys)} groups for test")

        # Prepare training and testing data based on allocated keys
        train_data = []
        test_data = []

        # Iterate through the flat list of all data items
        for item in all_data_items:
             if item["video_key"] in train_keys:
                 train_data.append(item["data"])
             elif item["video_key"] in test_keys:
                 test_data.append(item["data"])
             # Items with keys not in train_keys or test_keys (shouldn't happen with current logic) are skipped.


        # Record statistics for this subject
        stats["subjects"][subject] = {
            "train_items": len(train_data), # Number of individual step data items
            "test_items": len(test_data),
            "train_groups": len(train_keys), # Number of unique video groups
            "test_groups": len(test_keys)
        }

        # Update total statistics
        stats["total"]["train"] += len(train_data)
        stats["total"]["test"] += len(test_data)
        stats["total"]["train_groups"] += len(train_keys)
        stats["total"]["test_groups"] += len(test_keys)

        # Write training set JSONL file
        train_output_subject_dir = os.path.join(train_dir, subject)
        os.makedirs(train_output_subject_dir, exist_ok=True)
        train_output_file = os.path.join(train_output_subject_dir, f"{subject}_dataset_en.jsonl")
        try:
            with open(train_output_file, 'w', encoding='utf-8') as f:
                for item in train_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            logging.info(f"Wrote {len(train_data)} training data items to {train_output_file}")
        except IOError as e:
             logging.error(f"Error writing training file {train_output_file}: {e}")


        # Write testing set JSONL file
        test_output_subject_dir = os.path.join(test_dir, subject)
        os.makedirs(test_output_subject_dir, exist_ok=True)
        test_output_file = os.path.join(test_output_subject_dir, f"{subject}_dataset_en.jsonl")
        try:
            with open(test_output_file, 'w', encoding='utf-8') as f:
                for item in test_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            logging.info(f"Wrote {len(test_data)} testing data items to {test_output_file}")
        except IOError as e:
             logging.error(f"Error writing testing file {test_output_file}: {e}")


    # Save split statistics to a JSON file
    stats_output = os.path.join(output_dir, "dataset_stats.json")
    try:
        with open(stats_output, 'w', encoding='utf-8') as f:
            json.dump({
                "stats": stats,
                "ratios": f"{train_ratio}:{test_ratio}",
                "split_method": "group_based" # Add method info
            }, f, ensure_ascii=False, indent=4)

        logging.info(f"Dataset statistics saved to {stats_output}")
    except IOError as e:
        logging.error(f"Error writing stats file {stats_output}: {e}")


    logging.info("Dataset splitting complete!")

    return stats

def main():
    parser = argparse.ArgumentParser(description='Split JSONL dataset into training and testing sets based on video groups.')
    # Changed default paths to generic placeholders
    parser.add_argument('--source', type=str, default='/path/to/source_jsonl_data',
                        help='Source directory containing subject folders with JSONL files (e.g., /home/user/dataset/baseline_data/subject)')
    parser.add_argument('--output', type=str, default='/path/to/output_split_data',
                        help='Output directory for the split train/test data (e.g., /home/user/dataset/baseline_data/split)')
    parser.add_argument('--train_ratio', type=int, default=10,
                        help='Numerator for the training set ratio (e.g., 10 for 10:1 split)')
    parser.add_argument('--test_ratio', type=int, default=1,
                        help='Numerator for the testing set ratio (e.g., 1 for 10:1 split)')
    # Added argument for log directory base
    parser.add_argument('--log_base_dir', type=str, default="./logs",
                        help='Base directory for log files.')

    args = parser.parse_args()

    # Set up logging
    log_file = setup_logging(args.log_base_dir)
    logging.info(f"Starting dataset splitting. Source: {args.source}, Output: {args.output}, Ratio: {args.train_ratio}:{args.test_ratio}")

    # Split the dataset
    stats = split_dataset(args.source, args.output, args.train_ratio, args.test_ratio)

    # Print summary information
    logging.info(f"Dataset splitting complete! Total items: Train {stats['total']['train']}, Test {stats['total']['test']}")
    logging.info(f"Total groups: Train {stats['total']['train_groups']}, Test {stats['total']['test_groups']}")
    logging.info(f"Split data saved to: {args.output}")
    logging.info(f"Detailed log saved to: {log_file}")

    print(f"\nDataset splitting complete! Total items: Train {stats['total']['train']}, Test {stats['total']['test']}")
    print(f"Total groups: Train {stats['total']['train_groups']}, Test {stats['total']['test_groups']}")
    print(f"Split data saved to: {args.output}")
    print(f"Detailed log saved to: {log_file}")

if __name__ == "__main__":
    main()