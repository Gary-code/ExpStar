#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import random
import argparse
from datetime import datetime
from copy import deepcopy

def setup_logging():
    """Set up logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"update_merge_test_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def load_retrieval_data(retrieval_file):
    """Load retrieval data file"""
    with open(retrieval_file, 'r', encoding='utf-8') as f:
        retrieval_data = json.load(f)
    
    # Create mapping from video path to data
    video_map = {}
    for item in retrieval_data:
        video_path = item.get("video_path", "")
        if video_path:
            video_map[video_path] = item
    
    return video_map

def process_merge_test(merge_test_file, retrieval_data, output_file, less=False):
    """Process merge_test.jsonl file
    
    Args:
        merge_test_file: Input merge_test.jsonl file path
        retrieval_data: Retrieval data mapping
        output_file: Output file path
        less: Whether to select only one irrelevant document
    """
    processed_items = []
    total_items = 0
    expanded_items = 0
    relevant_items = 0
    irrelevant_items = 0
    
    with open(merge_test_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_items += 1
            item = json.loads(line)
            
            # Get video path
            video_path = item.get("videos", [""])[0]
            if not video_path:
                logging.warning(f"Video path not found, skipping item")
                processed_items.append(item)
                continue
            
            # Find corresponding retrieval data
            retrieval_item = retrieval_data.get(video_path)
            if not retrieval_item:
                logging.warning(f"No retrieval data found for video {video_path}, keeping original")
                processed_items.append(item)
                continue
            
            # Check if retrieval is needed
            if not retrieval_item.get("is_rag", False):
                logging.info(f"Video {video_path} does not need retrieval, keeping original")
                processed_items.append(item)
                continue
            
            # Get document list
            documents = retrieval_item.get("documents", [])
            if not documents:
                logging.warning(f"No retrieval documents for video {video_path}, keeping original")
                processed_items.append(item)
                continue
            
            # Split documents into relevant and irrelevant groups
            relevant_docs = []
            irrelevant_docs = []
            for doc in documents:
                relevance_score = doc.get("relevant", 0)
                if relevance_score >= 3:
                    relevant_docs.append(doc)
                else:
                    irrelevant_docs.append(doc)
            
            # Process relevant documents (keep all)
            for doc in relevant_docs:
                new_item = deepcopy(item)
                text = doc.get("text", "")
                
                # Update message content
                messages = new_item["messages"]
                for msg in messages:
                    if msg["role"] == "user" and msg["content"] == "<paragraph>passage</paragraph>":
                        msg["content"] = f"<paragraph>{text}</paragraph>"
                    elif msg["role"] == "assistant" and "[Is Relevant?]" in msg["content"]:
                        msg["content"] = msg["content"].replace("[Is Relevant?]", "[Relevant]")
                
                processed_items.append(new_item)
                expanded_items += 1
                relevant_items += 1
            
            # Process irrelevant documents
            if less and irrelevant_docs:
                # In less mode, randomly select one irrelevant document
                selected_docs = [random.choice(irrelevant_docs)]
            else:
                # In normal mode, use all irrelevant documents
                selected_docs = irrelevant_docs
            
            for doc in selected_docs:
                new_item = deepcopy(item)
                text = doc.get("text", "")
                
                # Update message content
                messages = new_item["messages"]
                for msg in messages:
                    if msg["role"] == "user" and msg["content"] == "<paragraph>passage</paragraph>":
                        msg["content"] = f"<paragraph>{text}</paragraph>"
                    elif msg["role"] == "assistant" and "[Is Relevant?]" in msg["content"]:
                        msg["content"] = msg["content"].replace("[Is Relevant?]", "[Irrelevant]")
                
                processed_items.append(new_item)
                expanded_items += 1
                irrelevant_items += 1
    
    # Write processed data
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    return total_items, expanded_items, relevant_items, irrelevant_items

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Update retrieval and relevance information in merge_test.jsonl file')
    parser.add_argument('--input', type=str, 
                        default='data/merge/merge_train.jsonl',
                        help='Input merge_test.jsonl file path')
    parser.add_argument('--retrieval', type=str,
                        default='data/rag/results/retrieval_results.json',
                        help='Retrieval data file path')
    parser.add_argument('--output', type=str,
                        default='data/merge/rel/merge_train_updated.jsonl',
                        help='Output file path')
    parser.add_argument('--less', action='store_true',
                        help='Whether to select only one irrelevant document')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting to update merge_test.jsonl file")
    logging.info(f"Less mode: {'enabled' if args.less else 'disabled'}")
    
    try:
        # Load retrieval data
        logging.info("Loading retrieval data file...")
        retrieval_data = load_retrieval_data(args.retrieval)
        logging.info(f"Successfully loaded {len(retrieval_data)} retrieval records")
        
        # Process merge_test file
        logging.info("Starting to process merge_test.jsonl file...")
        total_items, expanded_items, relevant_items, irrelevant_items = process_merge_test(
            args.input, retrieval_data, args.output, args.less
        )
        
        # Print processing results
        logging.info("=" * 50)
        logging.info("Processing completed!")
        logging.info(f"Original items: {total_items}")
        logging.info(f"Expanded items: {expanded_items}")
        logging.info(f"Relevant items: {relevant_items}")
        logging.info(f"Irrelevant items: {irrelevant_items}")
        logging.info(f"Output file: {args.output}")
        logging.info("=" * 50)
        
        print("\nProcessing completed!")
        print(f"Original items: {total_items}")
        print(f"Expanded items: {expanded_items}")
        print(f"Relevant items: {relevant_items}")
        print(f"Irrelevant items: {irrelevant_items}")
        print(f"Output file: {args.output}")
        print(f"Detailed log saved to: {log_file}")
        
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        print(f"\nProcessing failed! See log for details: {log_file}")
        raise

if __name__ == "__main__":
    main() 