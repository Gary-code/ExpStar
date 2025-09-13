#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
from datetime import datetime
import argparse

def setup_logging(log_base_dir="./logs"):
    """Set up logging."""
    # Use a configurable base directory for logs
    log_dir = os.path.join(log_base_dir, "merge_logs") # Use a subdirectory specific to merging
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"merge_datasets_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return log_file

def merge_jsonl_files(source_dir, output_file):
    """
    Merges all JSONL files within a directory into a single file.

    Args:
        source_dir: Source directory containing multiple JSONL files.
        output_file: Path for the output merged JSONL file.
    Returns:
        dict: Statistics including number of files merged and total lines written.
    """
    total_lines = 0
    files_merged = 0

    # Check if the source directory exists
    if not os.path.isdir(source_dir):
        logging.error(f"Source directory not found: {source_dir}")
        return {
             "files_merged": 0,
             "total_lines": 0
        }

    # Get all JSONL files in the source directory
    jsonl_files = [f for f in os.listdir(source_dir) if f.endswith(".jsonl")]

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir: # Check if output_file included a directory
        os.makedirs(output_dir, exist_ok=True)


    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for jsonl_file in sorted(jsonl_files):
                jsonl_path = os.path.join(source_dir, jsonl_file)
                line_count = 0

                logging.info(f"Processing file: {jsonl_file}")

                try:
                    with open(jsonl_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            try:
                                # Validate if the line is valid JSON
                                json.loads(line)
                                outfile.write(line)
                                line_count += 1
                            except json.JSONDecodeError:
                                logging.error(f"Skipping invalid JSON line in: {jsonl_path}")
                            except Exception as e:
                                logging.error(f"Error processing line in {jsonl_path}: {e}")

                    total_lines += line_count
                    files_merged += 1
                    logging.info(f"Merged {line_count} lines from {jsonl_file}")

                except IOError as e:
                     logging.error(f"Error reading source file {jsonl_path}: {e}")
                except Exception as e:
                     logging.error(f"An unexpected error occurred while processing {jsonl_path}: {e}")

    except IOError as e:
        logging.error(f"Error writing to output file {output_file}: {e}")
        return {
             "files_merged": files_merged, # Return counts up to the point of error
             "total_lines": total_lines
        }
    except Exception as e:
        logging.error(f"An unexpected error occurred during merging: {e}")
        return {
             "files_merged": files_merged, # Return counts up to the point of error
             "total_lines": total_lines
        }


    return {
        "files_merged": files_merged,
        "total_lines": total_lines
    }

def main():
    parser = argparse.ArgumentParser(description='Merges JSONL files from train and test split directories.')
    # Changed default paths to generic placeholders
    parser.add_argument('--train_dir', type=str, default='/path/to/split_train_data',
                        help='Directory containing training set JSONL files (e.g., /home/user/dataset/baseline_data/split/train)')
    parser.add_argument('--test_dir', type=str, default='/path/to/split_test_data',
                        help='Directory containing testing set JSONL files (e.g., /home/user/dataset/baseline_data/split/test)')
    parser.add_argument('--output_dir', type=str, default='/path/to/merged_output_data',
                        help='Output directory for the merged JSONL files (e.g., /home/user/dataset/baseline_data/merge)')
    # Added argument for log directory base
    parser.add_argument('--log_base_dir', type=str, default="./logs",
                        help='Base directory for log files.')

    args = parser.parse_args()

    # Set up logging
    log_file = setup_logging(args.log_base_dir)
    logging.info(f"Starting dataset merging. Train dir: {args.train_dir}, Test dir: {args.test_dir}, Output dir: {args.output_dir}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Merge training set files
    train_output = os.path.join(args.output_dir, "merge_train.jsonl")
    logging.info(f"Merging training files from {args.train_dir} to {train_output}")
    train_stats = merge_jsonl_files(args.train_dir, train_output)
    logging.info(f"Training set merging complete! Merged {train_stats['files_merged']} files, total {train_stats['total_lines']} lines.")

    # Merge testing set files
    test_output = os.path.join(args.output_dir, "merge_test.jsonl")
    logging.info(f"Merging testing files from {args.test_dir} to {test_output}")
    test_stats = merge_jsonl_files(args.test_dir, test_output)
    logging.info(f"Testing set merging complete! Merged {test_stats['files_merged']} files, total {test_stats['total_lines']} lines.")

    # Save merge statistics
    stats = {
        "train": train_stats,
        "test": test_stats,
        "train_output_path": os.path.abspath(train_output), # Use absolute path
        "test_output_path": os.path.abspath(test_output)   # Use absolute path
    }

    stats_output = os.path.join(args.output_dir, "merge_stats.json")
    try:
        with open(stats_output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)

        logging.info(f"Merge statistics saved to {stats_output}")
    except IOError as e:
         logging.error(f"Error writing merge stats file {stats_output}: {e}")

    logging.info(f"Detailed log saved to: {log_file}")

    print(f"\nDataset merging complete!")
    print(f"Training set: Merged {train_stats['files_merged']} files, total {train_stats['total_lines']} lines, saved to {train_output}")
    print(f"Testing set: Merged {test_stats['files_merged']} files, total {test_stats['total_lines']} lines, saved to {test_output}")
    print(f"Detailed log saved to: {log_file}")

if __name__ == "__main__":
    main()