#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import argparse
import logging
import re
from datetime import datetime

def setup_logging():
    """Configure logging system"""
    log_dir = os.path.expanduser("~/logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"add_annotations_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def process_subject(subject, unit_id, processed_base_dir="[PROCESSED_BASE_DIR]", 
                   annotation_base_dir="[ANNOTATION_BASE_DIR]"):
    """
    Add annotation information to JSON files for specific subject and unit
    
    Args:
        subject: Subject name (e.g. "biology", "chemistry")
        unit_id: Unit ID (e.g. "bio_35", "ch_94")
        processed_base_dir: Base directory for processed data
        annotation_base_dir: Base directory for annotation data
    
    Returns:
        tuple: (total_files, successfully_annotated_files, missing_annotation_files)
    """
    # Set paths
    processed_json_dir = os.path.join(processed_base_dir, subject, unit_id, "json")
    annotation_dir = os.path.join(annotation_base_dir, subject, unit_id, "asr")
    
    # Check if paths exist
    if not os.path.exists(processed_json_dir):
        logging.error(f"Processed JSON directory not found: {processed_json_dir}")
        return 0, 0, 0
    
    if not os.path.exists(annotation_dir):
        logging.error(f"Annotation directory not found: {annotation_dir}")
        return 0, 0, 0
    
    logging.info(f"Processing {subject}/{unit_id}")
    logging.info(f"Processed JSON directory: {processed_json_dir}")
    logging.info(f"Annotation directory: {annotation_dir}")
    
    # Get all unit directories
    unit_dirs = [d for d in os.listdir(processed_json_dir) if os.path.isdir(os.path.join(processed_json_dir, d))]
    logging.info(f"Found {len(unit_dirs)} unit directories")
    
    # Get all annotation files and build mapping (handling both underscore and hyphen cases)
    annotation_files = {}
    for ann_file in os.listdir(annotation_dir):
        if ann_file.endswith(".json"):
            base_name = os.path.splitext(ann_file)[0]
            # Store both original name and transformed name (underscore to hyphen)
            annotation_files[base_name] = os.path.join(annotation_dir, ann_file)
            dash_name = base_name.replace("_", "-")
            if dash_name != base_name:
                annotation_files[dash_name] = os.path.join(annotation_dir, ann_file)
    
    logging.info(f"Found {len(set(annotation_files.values()))} annotation files")
    
    # Statistics
    total_files = 0
    annotated_files = 0
    missing_annotation_files = 0
    
    for unit_dir in unit_dirs:
        unit_path = os.path.join(processed_json_dir, unit_dir)
        unit_name = unit_dir  # e.g. "01-Experiment1-UsingMicroscope"
        
        # Try multiple possible annotation file name matches
        annotation_file = None
        
        # 1. Direct match
        if unit_name in annotation_files:
            annotation_file = annotation_files[unit_name]
        # 2. Try replacing hyphen with underscore
        elif unit_name.replace("-", "_") in annotation_files:
            annotation_file = annotation_files[unit_name.replace("-", "_")]
        # 3. Extract number and name, ignoring format differences
        else:
            # Extract number and name parts
            match = re.match(r'^(\d+)[-_](.+)$', unit_name)
            if match:
                number, name = match.groups()
                # Try different format combinations
                for sep in ['_', '-']:
                    possible_name = f"{number}{sep}{name}"
                    if possible_name in annotation_files:
                        annotation_file = annotation_files[possible_name]
                        break
        
        if not annotation_file:
            logging.warning(f"No matching annotation file found, unit name: {unit_name}, skipping...")
            # Print available annotation filenames for debugging
            if len(annotation_files) < 10:  # Only print when annotation files are few
                logging.debug(f"Available annotation files: {list(annotation_files.keys())}")
            continue
        
        # Load annotation file
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)
        
        # Ensure annotation file has "subtitles" field
        if "subtitles" not in annotation_data:
            logging.warning(f"Annotation file {annotation_file} missing 'subtitles' field, skipping...")
            continue
        
        logging.info(f"Found annotation file for unit {unit_name}: {os.path.basename(annotation_file)}")
        
        # Group annotations by step
        step_annotations = {}
        for subtitle in annotation_data["subtitles"]:
            if "step" in subtitle:
                step_num = subtitle["step"]
                if step_num not in step_annotations:
                    step_annotations[step_num] = []
                
                # Extract required fields
                annotation_entry = {
                    "id": subtitle.get("id"),
                    "startTime": subtitle.get("startTime"),
                    "endTime": subtitle.get("endTime"),
                    "text": subtitle.get("text"),
                    "text_en": subtitle.get("text_en", "")
                }
                
                # Add optional fields if present
                for field in ["safety_en", "safety_zh", "equation", "principle_zh", "principle_en"]:
                    if field in subtitle:
                        annotation_entry[field] = subtitle[field]
                
                step_annotations[step_num].append(annotation_entry)
        
        # Get all step files for this unit
        step_files = glob.glob(os.path.join(unit_path, f"{unit_name}_step_*.json"))
        total_files += len(step_files)
        
        for step_file in step_files:
            with open(step_file, 'r', encoding='utf-8') as f:
                step_data = json.load(f)
            
            # Get step number
            step_number = step_data.get("step_number")
            if step_number is None:
                logging.warning(f"File {step_file} missing 'step_number' field, skipping...")
                continue
            
            # Find annotations for this step
            if step_number in step_annotations:
                # Add annotations field
                step_data["annotations"] = step_annotations[step_number]
                
                # Save updated JSON
                with open(step_file, 'w', encoding='utf-8') as f:
                    json.dump(step_data, f, ensure_ascii=False, indent=2)
                
                logging.info(f"Added {len(step_annotations[step_number])} annotations to {os.path.basename(step_file)}")
                annotated_files += 1
            else:
                logging.warning(f"No annotations found for step {step_number}, skipping {step_file}...")
                missing_annotation_files += 1
    
    logging.info(f"Processing complete! Total files: {total_files}, Annotated files: {annotated_files}, Missing annotations: {missing_annotation_files}")
    return total_files, annotated_files, missing_annotation_files

def process_all_subjects(processed_base_dir="[PROCESSED_BASE_DIR]", 
                         annotation_base_dir="[ANNOTATION_BASE_DIR]"):
    """Process all units across all subjects"""
    # Get all subjects
    subjects = [d for d in os.listdir(processed_base_dir) if os.path.isdir(os.path.join(processed_base_dir, d))]
    
    # Statistics
    total_stats = {
        "total_files": 0,
        "annotated_files": 0,
        "missing_annotation_files": 0,
        "processed_units": 0
    }
    
    for subject in subjects:
        subject_dir = os.path.join(processed_base_dir, subject)
        # Get all units for this subject
        unit_ids = [d for d in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, d))]
        
        for unit_id in unit_ids:
            total_files, annotated_files, missing_annotation_files = process_subject(
                subject, unit_id, processed_base_dir, annotation_base_dir
            )
            
            total_stats["total_files"] += total_files
            total_stats["annotated_files"] += annotated_files
            total_stats["missing_annotation_files"] += missing_annotation_files
            total_stats["processed_units"] += 1
    
    logging.info("=" * 50)
    logging.info("Processing summary:")
    logging.info(f"Processed units: {total_stats['processed_units']}")
    logging.info(f"Total files: {total_stats['total_files']}")
    logging.info(f"Annotated files: {total_stats['annotated_files']}")
    logging.info(f"Missing annotations: {total_stats['missing_annotation_files']}")
    logging.info("=" * 50)
    
    return total_stats

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Add annotation information to processed JSON files')
    parser.add_argument('--subject', type=str, help='Subject name (e.g. biology, chemistry)')
    parser.add_argument('--unit_id', type=str, help='Unit ID (e.g. bio_35, ch_94)')
    parser.add_argument('--processed_base_dir', type=str, 
                        default='[PROCESSED_BASE_DIR]',
                        help='Base directory for processed data')
    parser.add_argument('--annotation_base_dir', type=str, 
                        default='[ANNOTATION_BASE_DIR]',
                        help='Base directory for annotation data')
    parser.add_argument('--all', action='store_true', help='Process all subjects and units')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    
    # Start processing
    if args.all:
        logging.info("Processing all subjects and units")
        stats = process_all_subjects(args.processed_base_dir, args.annotation_base_dir)
    elif args.subject and args.unit_id:
        logging.info(f"Processing {args.subject}/{args.unit_id}")
        total_files, annotated_files, missing_annotation_files = process_subject(
            args.subject, args.unit_id, args.processed_base_dir, args.annotation_base_dir
        )
        stats = {
            "total_files": total_files,
            "annotated_files": annotated_files,
            "missing_annotation_files": missing_annotation_files,
            "processed_units": 1
        }
    else:
        logging.error("Must provide --subject and --unit_id parameters, or use --all to process all subjects and units")
        parser.print_help()
        return
    
    # Print summary
    print("\n" + "=" * 50)
    print("Processing summary:")
    print(f"Processed units: {stats['processed_units']}")
    print(f"Total files: {stats['total_files']}")
    print(f"Annotated files: {stats['annotated_files']}")
    print(f"Missing annotations: {stats['missing_annotation_files']}")
    print("=" * 50)
    print(f"Detailed logs saved to: {log_file}")

if __name__ == "__main__":
    main()
