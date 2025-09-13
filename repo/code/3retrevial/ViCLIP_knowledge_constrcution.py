import json
import torch
import cv2
import numpy as np
from tqdm import tqdm
from simple_tokenizer import SimpleTokenizer as _Tokenizer
from viclip import ViCLIP
import os

def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
def normalize(data):
    return (data/255.0-v_mean)/v_std

def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert(len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube

def get_vid_feat(frames, clip):
    return clip.get_vid_features(frames)

def get_text_feat(text, clip, tokenizer):
    return clip.get_text_features(text, tokenizer)

def main():
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = _Tokenizer()
    clip = ViCLIP(tokenizer).to(device)
    
    # Read knowledge base
    kb_path = 'data/knowledge_base/kb.jsonl'
    kb_docs = []
    with open(kb_path, 'r', encoding='utf-8') as f:
        for line in f:
            kb_docs.append(json.loads(line.strip()))
    
    # Pre-compute knowledge base text features
    print("Pre-computing knowledge base text features...")
    kb_texts = [doc['text'] for doc in kb_docs]
    kb_features = []
    for text in tqdm(kb_texts):
        feat = get_text_feat(text, clip, tokenizer)
        kb_features.append(feat)
    kb_features = torch.cat(kb_features, dim=0)
    
    # Read query file
    query_path = 'data/queries/video_title_nar_query.jsonl'
    queries = []
    with open(query_path, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line.strip()))
    
    # Process each query
    results = []
    for query in tqdm(queries):
        # Read video frames
        video = cv2.VideoCapture(query['video_path'])
        frames = [x for x in _frame_from_video(video)]
        video.release()
        
        # Get video features
        frames_tensor = frames2tensor(frames, device=device)
        vid_feat = get_vid_feat(frames_tensor, clip)
        
        # Get text features
        text_feat = get_text_feat(query['instruction'], clip, tokenizer)
        
        # Combine features (using video features only)
        combined_feat = 1.0 * vid_feat 
        
        # Calculate similarities
        similarities = torch.matmul(combined_feat, kb_features.T)
        top8_sim, top8_idx = torch.topk(similarities, k=8)
        
        # Get top 8 documents
        top8_docs = [kb_docs[i] for i in top8_idx[0].cpu().numpy()]
        
        # Save results
        result = {
            'query_id': query['id'],
            'query_instruction': query['instruction'],
            'query_video_path': query['video_path'],
            'top8_docs': top8_docs,
            'similarities': top8_sim[0].cpu().numpy().tolist()
        }
        results.append(result)
    
    # Save results
    output_path = 'outputs/viclip/video/kb.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_path}")

if __name__ == '__main__':
    main() 