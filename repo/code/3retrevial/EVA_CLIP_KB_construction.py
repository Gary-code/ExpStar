import json
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor, CLIPTokenizer
from tqdm import tqdm
import cv2
import numpy as np
from pathlib import Path
import os

def load_model():
    model_name_or_path = "models/EVA-CLIP-8B"
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to('cuda').eval()
    return model, processor, tokenizer

def extract_video_features(video_path, model, processor, fps=1):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    # Calculate sampling interval (in frames)
    frame_interval = int(video_fps / fps)
    
    frames = []
    frame_idx = 0
    
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:  # Sample one frame per second
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        
        frame_idx += 1
    
    cap.release()
    
    if not frames:
        return None
    
    # Process all frames
    inputs = processor(images=frames, return_tensors="pt", padding=True)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(inputs['pixel_values'])
        image_features = image_features.mean(dim=0)  # Average features of all frames
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    return image_features

def encode_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(inputs['input_ids'])
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return text_features

def main():
    # Load model
    print("Loading EVA-CLIP model...")
    model, processor, tokenizer = load_model()
    
    # Load query data
    print("Loading query data...")
    query_path = 'data/queries/video_title_nar_query.jsonl'
    with open(query_path, 'r', encoding='utf-8') as f:
        queries = [json.loads(line) for line in f]
    
    # Load knowledge base
    print("Loading knowledge base...")
    kb_path = 'data/knowledge_base/kb.jsonl'
    with open(kb_path, 'r', encoding='utf-8') as f:
        knowledge_base = [json.loads(line) for line in f]
    
    # Preprocess knowledge base text
    print("Preprocessing knowledge base text...")
    kb_texts = [f"{item['title']} {item['section']} {item['text']}" for item in knowledge_base]
    kb_features = []
    
    # Batch process knowledge base text
    batch_size = 32
    for i in tqdm(range(0, len(kb_texts), batch_size)):
        batch_texts = kb_texts[i:i+batch_size]
        batch_features = encode_text(batch_texts, model, tokenizer)
        kb_features.append(batch_features)
    
    kb_features = torch.cat(kb_features, dim=0)
    
    # Process each query
    results = []
    for query in tqdm(queries, desc="Processing queries"):
        # Extract video features
        video_features = extract_video_features(query['video_path'], model, processor)
        if video_features is None:
            print(f"Warning: Unable to process video {query['video_path']}")
            continue
        
        # Extract text features
        text_features = encode_text(query['instruction'], model, tokenizer)
        
        # Combine features (using video features only)
        combined_features = 1.0 * video_features + 0.0 * text_features
        combined_features /= combined_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarities = (100.0 * combined_features @ kb_features.T).squeeze()
        
        # Get top-8 results
        top_k = 8
        top_indices = torch.topk(similarities, k=top_k).indices
        
        # Build result
        result = {
            "id": query['id'],
            "query": query['instruction'],
            "video_path": query['video_path'],
            "documents": [knowledge_base[idx] for idx in top_indices.tolist()]
        }
        results.append(result)
    
    # Save results
    output_path = 'outputs/eva-clip/video/kb.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main() 