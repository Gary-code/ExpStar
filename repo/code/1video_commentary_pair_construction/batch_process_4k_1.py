#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Batch process 4K videos and step files')
    parser.add_argument('--base_dir', type=str, default='/path/to/base_dir/4k_0', help='Source base directory')
    parser.add_argument('--output_base', type=str, default='/path/to/output_base/4k_1', help='Output base directory')
    parser.add_argument('--process_all', action='store_true', help='Process all subjects and units')
    parser.add_argument('--subject', type=str, help='Subject name to process all its units')
    
    args = parser.parse_args()
    
    # Create log directory
    log_dir = os.path.join(args.output_base, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"batch_process_{timestamp}.log")
    
    # Collect subject-unit pairs
    subjects_units = []
    
    # If processing all subjects and units, scan the dataset directory structure
    if args.process_all:
        dataset_dir = os.path.join(args.base_dir, "dataset")
        if os.path.exists(dataset_dir):
            for subject in os.listdir(dataset_dir):
                subject_dir = os.path.join(dataset_dir, subject)
                if os.path.isdir(subject_dir):
                    for unit_id in os.listdir(subject_dir):
                        unit_dir = os.path.join(subject_dir, unit_id)
                        if os.path.isdir(unit_dir) and os.path.exists(os.path.join(unit_dir, "step")):
                            subjects_units.append((subject, unit_id))
    
    # If a specific subject is provided, only process its units
    elif args.subject:
        subject_dir = os.path.join(args.base_dir, "dataset", args.subject)
        if os.path.exists(subject_dir):
            for unit_id in os.listdir(subject_dir):
                unit_dir = os.path.join(subject_dir, unit_id)
                if os.path.isdir(unit_dir) and os.path.exists(os.path.join(unit_dir, "step")):
                    subjects_units.append((args.subject, unit_id))
    
    # Exit if no valid subject-unit pairs found
    if not subjects_units:
        print("No subject-unit combinations found to process")
        return
    
    print(f"Found {len(subjects_units)} subject-unit combinations. Starting processing...")
    print(f"Log file: {log_file}")
    
    # Process each subject-unit pair
    results = {}
    for subject, unit_id in subjects_units:
        print(f"\nProcessing: {subject}/{unit_id}")
        
        # Build command
        cmd = [
            "python3", "/path/to/process_videos_4k_1.py",
            "--subject", subject,
            "--unit_id", unit_id,
            "--base_dir", args.base_dir,
            "--output_base", args.output_base
        ]
        
        try:
            # Execute command
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Log output
            with open(log_file, 'a') as f:
                f.write(f"{'='*50}\n")
                f.write(f"Processing: {subject}/{unit_id}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"STDOUT:\n{result.stdout}\n")
                f.write(f"STDERR:\n{result.stderr}\n")
                f.write(f"{'='*50}\n\n")
            
            if result.returncode == 0:
                print(f"Successfully processed: {subject}/{unit_id}")
                results[(subject, unit_id)] = "Success"
            else:
                print(f"Failed to process: {subject}/{unit_id}")
                results[(subject, unit_id)] = "Failure"
                
        except Exception as e:
            print(f"Exception occurred while processing: {subject}/{unit_id} - {str(e)}")
            results[(subject, unit_id)] = f"Error: {str(e)}"
            
            # Log exception
            with open(log_file, 'a') as f:
                f.write(f"{'='*50}\n")
                f.write(f"Exception occurred: {subject}/{unit_id}\n")
                f.write(f"Error message: {str(e)}\n")
                f.write(f"{'='*50}\n\n")
    
    # Print summary
    print("\n\nProcessing Summary:")
    print(f"Total: {len(subjects_units)}")
    print(f"Success: {list(results.values()).count('Success')}")
    print(f"Failed: {len(subjects_units) - list(results.values()).count('Success')}")
    print(f"See detailed logs at: {log_file}")

if __name__ == "__main__":
    main()