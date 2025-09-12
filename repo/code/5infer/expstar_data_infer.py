#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import re
import time
import argparse
import logging
import jsonlines
from datetime import datetime
from swift.llm import InferRequest, InferClient, RequestConfig
from swift.plugin import InferStats
import random

def setup_logging(output_dir):
    """Configure logging settings"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    logging.info(f"Log file will be saved to: {log_file}")

def list_to_jsonl(result_list:list, output_path:str):
    """Write result list to JSONL file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in result_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_retrieval_data(retrieval_file):
    """Load retrieval data file"""
    logging.info(f"Loading retrieval data file: {retrieval_file}")
    with open(retrieval_file, 'r', encoding='utf-8') as f:
        retrieval_data = json.load(f)
    
    logging.info(f"Successfully loaded retrieval data, total {len(retrieval_data)} records")
    
    # Create mapping from video path to data
    video_map = {}
    for item in retrieval_data:
        video_path = item.get("video_path", "")
        if video_path:
            video_path = os.path.normpath(video_path)
            video_map[video_path] = item
    
    logging.info(f"Successfully created video path mapping, total {len(video_map)} valid paths")
    return video_map

def log_error(error_info, item_id, error_type, details=None):
    """Function to log error information"""
    error_log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "item_id": item_id,
        "error_type": error_type,
        "error_message": str(error_info),
        "details": details
    }
    logging.error(json.dumps(error_log, ensure_ascii=False))
    return error_log

def simulate_rag_retrieval(video_path, retrieval_data, num_samples=3):
    """Perform RAG retrieval"""
    video_path = os.path.normpath(video_path)
    logging.info(f"Attempting to retrieve video: {video_path}")
    
    retrieval_item = retrieval_data.get(video_path)
    if not retrieval_item:
        logging.warning(f"No retrieval data found for video")
        return []
    
    documents = retrieval_item.get("documents", [])
    if not documents:
        logging.warning("No retrieval documents for video")
        return []
    
    logging.info(f"Found {len(documents)} documents")
    return documents[:num_samples]

def truncate_response(content, max_length=1000):
    """Truncate overly long responses"""
    if len(content) <= max_length:
        return content
    
    sentences = re.split(r'([。！？.!?])', content[:max_length])
    if len(sentences) >= 2:
        truncated = ''.join(sentences[:-2] + sentences[-2:-1])
        return truncated
    return content[:max_length]

def process_item(item, engine_pool, request_config, metric, retrieval_data=None, num_samples=8, max_retries=3):
    """Function to process a single item, with retry mechanism and detailed output information"""
    n = random.randint(0, len(engine_pool) - 1)
    engine = engine_pool[n]
    
    item_id = item.get("id", f"item_{hash(str(item))}")
    logging.info(f"Processing sample ID: {item_id}")
    
    try:
        # Extract system, user, and assistant messages
        system_messages = [msg for msg in item["messages"] if msg["role"] == "system"]
        user_messages = [msg for msg in item["messages"] if msg["role"] == "user"]
        assistant_messages = [msg for msg in item["messages"] if msg["role"] == "assistant"]
        
        if not system_messages:
            error_info = "No system message found in the item"
            log_error(error_info, item_id, "MISSING_SYSTEM_MESSAGE")
            raise ValueError(error_info)
            
        if not user_messages:
            error_info = "No user message found in the item"
            log_error(error_info, item_id, "MISSING_USER_MESSAGE")
            raise ValueError(error_info)
        
        if not assistant_messages:
            error_info = "No assistant message (ground truth) found in the item"
            log_error(error_info, item_id, "MISSING_ASSISTANT_MESSAGE")
            raise ValueError(error_info)
        
        logging.info("Message extraction completed")
        
        # Extract user input
        input_query = user_messages[0]["content"]
        logging.info("User input extracted")
        
        # Get video path
        if not item.get("videos"):
            error_info = "No video path found in the item"
            log_error(error_info, item_id, "MISSING_VIDEO_PATH")
            raise ValueError(error_info)
        
        video_path = item["videos"][0] if isinstance(item["videos"], list) else item["videos"]
        logging.info(f"Video path: {video_path}")
        
        # Check if file exists
        if not os.path.exists(video_path):
            error_info = f"Video file not found: {video_path}"
            log_error(error_info, item_id, "VIDEO_FILE_NOT_FOUND")
            raise FileNotFoundError(error_info)
        
        logging.info("File check completed")
        
        # Build first round message with system prompt
        first_round_message = [
            {'role': 'system', 'content': system_messages[0]["content"]},
            {'role': 'user', 'content': input_query}
        ]
        
        logging.info("Starting first round conversation")
        in_video_path = [video_path]
        
        # First round request (with retry)
        for retry in range(max_retries):
            try:
                if retry > 0:
                    logging.info(f"First round request retry ({retry+1}/{max_retries})")
                infer_request = [InferRequest(
                    messages=first_round_message,
                    videos=in_video_path 
                )]
                first_round_response = engine.infer(infer_request, request_config, metrics=[metric])
                first_round_content = first_round_response[0].choices[0].message.content
                logging.info("First round response received")
                break
            except Exception as e:
                error_info = f"First round request failed: {str(e)}"
                if retry == max_retries - 1:
                    log_error(error_info, item_id, "FIRST_ROUND_API_ERROR", {"retry_count": retry + 1})
                    raise Exception(error_info)
                time.sleep(1)
        
        # Check if retrieval is needed
        retrieval_detected = "[Retrieval]" in first_round_content
        retrieval_index = first_round_content.find("[Retrieval]") if retrieval_detected else -1
        
        evaluation_data = {
            "rounds": [],
            "gt": item.get("messages", []),
            "is_retrieval": retrieval_detected,
            "rag_passages": [],
            "relevant_passages": []
        }
        
        if retrieval_detected:
            truncated_content = first_round_content[:retrieval_index] + "[Retrieval]"
            evaluation_data["rounds"].append({"role": "assistant", "content": truncated_content})
            
            # Perform retrieval
            logging.info("Performing retrieval")
            retrieved_documents = simulate_rag_retrieval(video_path, retrieval_data, num_samples)
            
            if not retrieved_documents:
                logging.warning("No documents retrieved")
                evaluation_data["rag_passages"] = []
            else:
                logging.info(f"Retrieved {len(retrieved_documents)} documents")
                evaluation_data["rag_passages"] = retrieved_documents
            
            # Build second round message with retrieved documents
            second_round_message = [
                {'role': 'system', 'content': system_messages[0]["content"]},
                {'role': 'user', 'content': input_query},
                {'role': 'assistant', 'content': truncated_content},
                {'role': 'user', 'content': f"Here are the retrieved documents:\n{json.dumps(retrieved_documents, ensure_ascii=False, indent=2)}"}
            ]
            
            logging.info("Starting second round conversation")
            
            # Second round request (with retry)
            for retry in range(max_retries):
                try:
                    if retry > 0:
                        logging.info(f"Second round request retry ({retry+1}/{max_retries})")
                    infer_request = [InferRequest(
                        messages=second_round_message,
                        videos=in_video_path 
                    )]
                    second_round_response = engine.infer(infer_request, request_config, metrics=[metric])
                    second_round_content = second_round_response[0].choices[0].message.content
                    logging.info("Second round response received")
                    break
                except Exception as e:
                    error_info = f"Second round request failed: {str(e)}"
                    if retry == max_retries - 1:
                        log_error(error_info, item_id, "SECOND_ROUND_API_ERROR", {"retry_count": retry + 1})
                        raise Exception(error_info)
                    time.sleep(1)
            
            evaluation_data["rounds"].append({"role": "assistant", "content": second_round_content})
        else:
            evaluation_data["rounds"].append({"role": "assistant", "content": first_round_content})
        
        logging.info("Processing completed successfully")
        return evaluation_data
        
    except Exception as e:
        error_info = f"Error processing item: {str(e)}"
        log_error(error_info, item_id, "PROCESSING_ERROR")
        raise Exception(error_info)

def main():
    parser = argparse.ArgumentParser(description='RAG Inference Script')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file path')
    parser.add_argument('--retrieval', type=str, help='Retrieval data file path')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of samples to retrieve')
    parser.add_argument('--max_retries', type=int, default=3, help='Maximum number of retries for API calls')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Setup logging
    output_dir = os.path.dirname(args.output)
    setup_logging(output_dir)
    
    # Load retrieval data if provided
    retrieval_data = None
    if args.retrieval:
        retrieval_data = load_retrieval_data(args.retrieval)
    
    # Initialize engine pool
    engine_pool = [InferClient() for _ in range(1)]
    request_config = RequestConfig()
    metric = InferStats()
    
    # Process items
    results = []
    with jsonlines.open(args.input, mode='r') as reader:
        items = list(reader)
    
    logging.info(f"Processing {len(items)} items...")
    for i in range(0, len(items), args.batch_size):
        batch = items[i:i + args.batch_size]
        for item in batch:
            try:
                result = process_item(
                    item,
                    engine_pool,
                    request_config,
                    metric,
                    retrieval_data,
                    args.num_samples,
                    args.max_retries
                )
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing item: {str(e)}")
                continue
    
    # Save results
    list_to_jsonl(results, args.output)
    logging.info(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main() 