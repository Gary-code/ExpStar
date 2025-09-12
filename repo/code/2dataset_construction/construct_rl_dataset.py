import json
import os
from tqdm import tqdm
from collections import defaultdict

def load_train_data():
    """Load training data to get system prompt and user prompt"""
    train_file = "/path/to/dataset/merge_train.jsonl"
    system_prompts = {}
    user_prompts = {}
    video_paths = {}
    print("Loading training data...")
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if 'id' in item and 'messages' in item:
                video_paths[item['id']] = item.get('videos', [])[0]
                for msg in item['messages']:
                    if msg['role'] == 'system':
                        system_prompts[item['id']] = msg['content']
                    if msg['role'] == 'user':
                        user_prompts[item['id']] = msg['content']
                        break
    
    return system_prompts, user_prompts, video_paths

def construct_rl_dataset():
    # Set file paths
    input_file = "/path/to/results/merged_with_reward_cleaned_filtered.jsonl"
    output_file = input_file.replace('.jsonl', '_rl.jsonl')
    
    print(f"Processing file: {input_file}")
    
    # Load training data to get system prompt
    system_prompts, user_prompts, video_paths = load_train_data()
    
    # Read data
    items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line.strip()))
    
    print(f"Read {len(items)} items")
    
    # Group by id
    id_groups = defaultdict(list)
    for item in items:
        id_groups[item['id']].append(item)
    
    print(f"Grouped into {len(id_groups)} groups by id")
    
    # Construct RL dataset
    rl_items = []
    for id, group in tqdm(id_groups.items(), desc="Processing groups"):
        # Get system prompt
        system_prompt = system_prompts.get(id, "You are a helpful AI assistant")
        
        # Separate positive and negative examples
        positive_items = [item for item in group if item['reward']]
        negative_items = [item for item in group if not item['reward']]
        
        # Get video path
        video_path = video_paths.get(id, None)
        
        # Construct data pairs
        for pos_item in positive_items:
            for neg_item in negative_items:
                rl_item = {
                    "id": id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompts[id]},
                        {"role": "assistant", "content": pos_item['response']},
                    ],
                    "videos": [video_path],
                    "rejected_response": neg_item['response']
                }
                rl_items.append(rl_item)
    
    # Save processed file
    print(f"\nSaving processed file to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(rl_items, desc="Saving data"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nProcessing completed!")
    print(f"Original data count: {len(items)}")
    print(f"Constructed RL pairs count: {len(rl_items)}")
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    construct_rl_dataset() 