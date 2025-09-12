#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import argparse
import logging
from datetime import datetime

def setup_logging():
    """Set up logging"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"create_finetune_dataset_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def get_system_message(subject):
    """
    Get system message based on subject
    
    Args:
        subject: subject name
    """
    if subject == "chemistry":
        msg = "You are an expert narrator for chemistry experiment videos, helping students better understand chemistry experiments, principles, and operational procedures. "
    elif subject == "physics":
        msg = "You are an expert narrator for physics experiment videos, helping students better understand physical phenomena, principles, and key operational points. "
    elif subject == "biology":
        msg = "You are an expert narrator for biology experiment videos, helping students better understand biological concepts, experimental processes, and observational results. "
    elif subject == "kid":
        msg = "You are a friendly narrator for children's fun science experiment videos, using simple and engaging language to explain scientific concepts to children aged 6-12, inspiring their interest and curiosity in science. "
    elif subject == "electronic":
        msg = "You are an expert narrator for electronics experiment videos, helping learners better understand electronic circuits, principles, and assembly and debugging points. "
    elif subject == "civil":
        msg = "You are an expert narrator for civil engineering experiment videos, helping learners better understand building structures, material properties, and engineering principles. "
    elif subject == "material":
        msg = "You are an expert narrator for materials science experiment videos, helping learners better understand material properties, structures, and application characteristics. "
    else:
        msg = "You are an expert narrator for science experiment videos, helping viewers better understand experimental content and scientific principles. "
    
    # Add tag usage instructions
    msg += "Always use <Narration> tags to mark the narration content. "
    msg += "When necessary, please: 1. Use <Principle> tags to mark explanations of scientific principles or theories. 2. Use <Safety> tags to mark safety precautions or warnings. "
    
    return msg

def get_prompt_template(subject, title):
    """
    Get prompt template based on subject
    
    Args:
        subject: subject name
        title: experiment title
    """
    if subject == "kid":
        return f"<video>Please generate a lively and fun narration for the current step of the children's science experiment titled \"{title}\".\n\n"
    elif subject == "chemistry":
        return f"<video>Please generate a detailed narration for the current step of the chemistry experiment titled \"{title}\".\n\n"
    elif subject == "physics":
        return f"<video>Please generate a detailed narration for the current step of the physics experiment titled \"{title}\".\n\n"
    elif subject == "biology":
        return f"<video>Please generate a detailed narration for the current step of the biology experiment titled \"{title}\".\n\n"
    elif subject == "electronic":
        return f"<video>Please generate a detailed narration for the current step of the electronics experiment titled \"{title}\".\n\n"
    elif subject == "civil":
        return f"<video>Please generate a detailed narration for the current step of the civil engineering experiment titled \"{title}\".\n\n"
    elif subject == "material":
        return f"<video>Please generate a detailed narration for the current step of the materials science experiment titled \"{title}\".\n\n"
    else:
        return f"<video>Please generate a detailed narration for the current step of the science experiment titled \"{title}\".\n\n"

def create_fine_tuning_dataset(source_subject, source_unit, 
                              source_base_dir="data/processed", 
                              output_base_dir="data/baseline_data/subject"):
    """
    Create fine-tuning dataset from source JSON files
    
    Args:
        source_subject: subject name (chemistry, physics, biology, kid, etc.)
        source_unit: unit ID (ch_81, phy_45, etc.)
        source_base_dir: base directory for source data
        output_base_dir: base path for output directory
    """
    # Define special tags
    NARRATION_START = "<Narration>\n"
    PRINCIPLE_START = "<Principle>\n"
    SAFETY_START = "<Safety>\n"
    
    # Get system message
    system_message = get_system_message(source_subject)
    
    # Set source and output paths
    source_json_dir = os.path.join(source_base_dir, source_subject, source_unit, "json")
    output_dir = os.path.join(output_base_dir, source_subject, source_unit)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if source directory exists
    if not os.path.exists(source_json_dir):
        logging.error(f"Source JSON directory does not exist: {source_json_dir}")
        return 0
    
    # Get all experiment directories
    experiment_dirs = [d for d in os.listdir(source_json_dir) 
                      if os.path.isdir(os.path.join(source_json_dir, d))]
    
    logging.info(f"Found {len(experiment_dirs)} experiment directories in {source_subject}/{source_unit}")
    
    # Store fine-tuning dataset
    fine_tuning_data = []
    
    for exp_dir in experiment_dirs:
        # Get all JSON files in the experiment directory
        json_pattern = os.path.join(source_json_dir, exp_dir, "*.json")
        json_files = sorted(glob.glob(json_pattern))
        
        # Store step summaries for each experiment
        experiment_steps_summary = {}
        
        # First pass: collect all step information
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
            
            step_number = source_data.get("step_number", 0)
            step_en = source_data.get("step_en", "")
            
            # Store step summary
            experiment_steps_summary[step_number] = {"step_en": step_en}
        
        # Second pass: generate narration with previous step summaries
        for json_file in json_files:
            # Read source JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
            
            # Get necessary fields
            title_en = source_data.get("title_en", "")
            video_path = source_data.get("clip_video_path", "")
            step_en = source_data.get("step_en", "")
            step_number = source_data.get("step_number", 0)
            
            # Get annotations (if any)
            annotations = source_data.get("annotations", [])
            
            # Get prompt template based on subject
            assistant_content = get_prompt_template(source_subject, title_en)
            
            # If not the first step, add summaries of previous steps
            if step_number > 1:
                assistant_content += "In previous steps:\n"
                for prev_step in range(1, step_number):
                    if prev_step in experiment_steps_summary:
                        assistant_content += f"- Step {prev_step}: {experiment_steps_summary[prev_step]['step_en']}\n"
                assistant_content += f"\nNow proceeding with Step {step_number}:\n"
            
            # Collect content according to required format
            narration_text = []
            principle_text = []
            safety_text = []
            
            if annotations:
                # Sort annotations by time
                sorted_annotations = sorted(annotations, key=lambda x: x.get("startTime", "0:00:00"))
                
                # Collect all text content
                for annotation in sorted_annotations:
                    text = annotation.get("text_en", "")
                    if text:
                        narration_text.append(text)
                    
                    # Collect safety information
                    if "safety_en" in annotation and annotation["safety_en"]:
                        safety_note = annotation["safety_en"]
                        if safety_note not in safety_text:
                            safety_text.append(safety_note)
                    
                    # Collect principle information
                    if "principle_en" in annotation and annotation["principle_en"]:
                        principle_note = annotation["principle_en"]
                        if principle_note not in principle_text:
                            principle_text.append(principle_note)
            
            gt_content = ""

            # Build final output content
            if narration_text:
                gt_content += NARRATION_START
                gt_content += "\n".join(narration_text) + "\n"
            
            if principle_text:
                gt_content += PRINCIPLE_START
                for note in principle_text:
                    gt_content += f"{note}\n"
            
            if safety_text:
                gt_content += SAFETY_START
                for note in safety_text:
                    gt_content += f"{note}\n"
            
            if gt_content == "":
                continue
            
            # Create fine-tuning data item
            fine_tuning_item = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": assistant_content
                    },
                    {
                        "role": "assistant",
                        "content": gt_content
                    }
                ],
                "videos": [video_path],
                "id": f"{source_unit}_{len(fine_tuning_data)}"
            }
            
            fine_tuning_data.append(fine_tuning_item)
    
    # Write fine-tuning dataset to output file
    output_file = os.path.join(output_dir, f"{source_unit}_dataset_en.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in fine_tuning_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logging.info(f"Successfully created fine-tuning dataset with {len(fine_tuning_data)} items, saved to {output_file}")
    return len(fine_tuning_data)

def process_all_subjects(source_base_dir="data/processed", 
                         output_base_dir="data/baseline_data/subject"):
    """
    Process all units under all subjects
    
    Args:
        source_base_dir: base directory for source data
        output_base_dir: base path for output directory
    """
    # Get all subjects
    subjects = [d for d in os.listdir(source_base_dir) if os.path.isdir(os.path.join(source_base_dir, d))]
    
    # Statistics
    total_items = 0
    processed_units = 0
    
    for subject in subjects:
        subject_dir = os.path.join(source_base_dir, subject)
        # Get all units under this subject
        unit_ids = [d for d in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, d))]
        
        for unit_id in unit_ids:
            logging.info(f"Processing {subject}/{unit_id}...")
            items_count = create_fine_tuning_dataset(
                subject, unit_id, source_base_dir, output_base_dir
            )
            
            total_items += items_count
            processed_units += 1
            
            logging.info(f"Completed processing {subject}/{unit_id}, generated {items_count} items")
    
    logging.info("=" * 50)
    logging.info("Processing Summary:")
    logging.info(f"Units processed: {processed_units}")
    logging.info(f"Total dataset items generated: {total_items}")
    logging.info("=" * 50)
    
    return {"total_items": total_items, "processed_units": processed_units}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create fine-tuning dataset from processed JSON files')
    parser.add_argument('--subject', type=str, help='Subject name (e.g., biology, chemistry)')
    parser.add_argument('--unit_id', type=str, help='Unit ID (e.g., bio_35, ch_94)')
    parser.add_argument('--source_base_dir', type=str, 
                        default='data/processed',
                        help='Base directory for source data')
    parser.add_argument('--output_base_dir', type=str, 
                        default='data/baseline_data/subject',
                        help='Base path for output directory')
    parser.add_argument('--all', action='store_true', help='Process all subjects and units')
    
    args = parser.parse_args()
    
    # Set up logging
    log_file = setup_logging()
    
    # Start processing
    if args.all:
        logging.info("Starting to process all subjects and units")
        stats = process_all_subjects(args.source_base_dir, args.output_base_dir)
        all_stats = stats
    elif args.subject and args.unit_id:
        logging.info(f"Starting to process {args.subject}/{args.unit_id}")
        items_count = create_fine_tuning_dataset(
            args.subject, args.unit_id, args.source_base_dir, args.output_base_dir
        )
        all_stats = {"total_items": items_count, "processed_units": 1}
    else:
        logging.error("Must provide --subject and --unit_id parameters, or use --all to process all subjects and units")
        parser.print_help()
        return
    
    # Print summary
    print("\n" + "=" * 50)
    print("Processing Summary:")
    print(f"Units processed: {all_stats['processed_units']}")
    print(f"Total dataset items generated: {all_stats['total_items']}")
    print("=" * 50)
    print(f"Detailed log saved to: {log_file}")

if __name__ == "__main__":
    main()