#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified Video-Text Joint Retrieval System
Processes jsonl format query files, handling all functions from embedding to retrieval
"""

import os
import sys
import json
import argparse
import time
import glob
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# Add current directory to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules
import src.index
import src.contriever
import src.data
import src.normalize_text

def ensure_openai_clip():
    """Ensure OpenAI CLIP model is installed and loaded
    
    Returns:
        CLIP model and preprocessing function
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        import clip
        if hasattr(clip, 'available_models') and callable(getattr(clip, 'load', None)):
            print("OpenAI CLIP detected, loading...")
            model, preprocess = clip.load("ViT-B/32", device=device)
            return model, preprocess
        else:
            raise ImportError("Imported clip is not OpenAI's CLIP")
    except (ImportError, AttributeError) as e:
        print(f"OpenAI CLIP not found or import error: {e}")
        print("Attempting to install OpenAI CLIP...")
        os.system("pip uninstall -y clip")
        os.system("pip install git+https://github.com/openai/CLIP.git")
        import sys
        if "clip" in sys.modules:
            del sys.modules["clip"]
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, preprocess

def load_local_model(model_path):
    """Load local model
    
    Args:
        model_path: Path to model
        
    Returns:
        Model, tokenizer and config
    """
    print(f"Loading local model: {model_path}")
    try:
        model, tokenizer, cfg = src.contriever.load_retriever(model_path)
        return model, tokenizer, cfg
    except Exception as e:
        print(f"Standard loading failed: {e}")
        
        import transformers
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model path not found: {model_path}")
            
        config = transformers.AutoConfig.from_pretrained(
            model_path, 
            local_files_only=True,
            trust_remote_code=True
        )
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, 
            local_files_only=True,
            trust_remote_code=True
        )
        
        model = transformers.AutoModel.from_pretrained(
            model_path, 
            config=config,
            local_files_only=True,
            trust_remote_code=True
        )
        
        return model, tokenizer, config

class VideoProcessor:
    """Video processor for frame extraction and feature generation"""
    
    def __init__(self, fps=1, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize video processor
        
        Args:
            fps: Frames per second to extract
            device: Computation device
        """
        self.fps = fps
        self.device = device
        
        print(f"Loading CLIP model...")
        self.clip_model, self.clip_preprocess = ensure_openai_clip()
        
    def extract_frames(self, video_path):
        """Extract frames from video
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of extracted frames
        """
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return []
            
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return []
            
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        sample_interval = int(video_fps / self.fps)
        if sample_interval < 1:
            sample_interval = 1
            
        print(f"Video FPS: {video_fps}, Total frames: {frame_count}, Sample interval: {sample_interval}")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
            frame_idx += 1
            
        cap.release()
        print(f"Extracted {len(frames)} frames from video")
        return frames
        
    def process_frames(self, frames):
        """Process extracted frames into CLIP features
        
        Args:
            frames: List of frames
            
        Returns:
            Frame feature vectors
        """
        if not frames:
            return None
            
        processed_images = []
        for frame in frames:
            pil_image = Image.fromarray(frame)
            processed_image = self.clip_preprocess(pil_image)
            processed_images.append(processed_image)
            
        image_batch = torch.stack(processed_images).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_batch)
            pooled_features = torch.max(image_features, dim=0)[0]
            pooled_features = pooled_features.float()
        
        return pooled_features
        
    def process_video(self, video_path):
        """Process video to extract frames and generate features
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video feature vector
        """
        frames = self.extract_frames(video_path)
        if not frames:
            return None
            
        features = self.process_frames(frames)
        return features


class SimpleVideoTextRetriever:
    """Simplified video-text joint retriever"""
    
    def __init__(self, model_path, passages_path, passages_embeddings_path, 
                 video_fps=1.0, text_weight=0.7, n_docs=10, use_fp16=True):
        """Initialize retriever
        
        Args:
            model_path: Path to text model
            passages_path: Path to documents file
            passages_embeddings_path: Path to document embeddings (glob pattern)
            video_fps: Video processing frame rate
            text_weight: Weight for text embedding (0-1)
            n_docs: Number of documents to retrieve per query
            use_fp16: Whether to use half-precision inference
        """
        self.model_path = model_path
        self.passages_path = passages_path
        self.passages_embeddings_path = passages_embeddings_path
        self.video_fps = video_fps
        self.text_weight = text_weight
        self.n_docs = n_docs
        self.use_fp16 = use_fp16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.video_processor = VideoProcessor(fps=video_fps, device=self.device)
        
        print(f"Loading text model: {model_path}")
        self.text_model, self.tokenizer, _ = load_local_model(model_path)
        self.text_model.eval()
        self.text_model = self.text_model.to(self.device)
        if use_fp16:
            self.text_model = self.text_model.half()
            
        self.index = src.index.Indexer(768, 0, 8)
        
        self.setup_retriever()
        
    def setup_retriever(self):
        """Setup retriever by loading index and documents"""
        input_paths = glob.glob(self.passages_embeddings_path)
        input_paths = sorted(input_paths)
        
        if not input_paths:
            raise ValueError(f"No embedding files found: {self.passages_embeddings_path}")
            
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        
        if os.path.exists(index_path):
            print(f"Loading existing index: {index_path}")
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Building index from files: {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(input_paths, 1000000)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} sec")
            self.index.serialize(embeddings_dir)

        print(f"Loading documents: {self.passages_path}")
        self.passages = src.data.load_passages(self.passages_path)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print(f"Loaded {len(self.passages)} documents")
        
    def index_encoded_data(self, embedding_files, indexing_batch_size):
        """Add precomputed embeddings to index
        
        Args:
            embedding_files: List of files containing embeddings
            indexing_batch_size: Batch size for indexing
        """
        allids = []
        allembeddings = np.array([])
        for i, file_path in enumerate(embedding_files):
            print(f"Loading file: {file_path}")
            with open(file_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)

            allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
            allids.extend(ids)
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = self.add_embeddings(allembeddings, allids, indexing_batch_size)

        while allembeddings.shape[0] > 0:
            allembeddings, allids = self.add_embeddings(allembeddings, allids, indexing_batch_size)

        print("Index building complete")
        
    def add_embeddings(self, embeddings, ids, indexing_batch_size):
        """Add batch of embeddings to index
        
        Args:
            embeddings: Array of embeddings
            ids: List of document IDs
            indexing_batch_size: Batch size for indexing
            
        Returns:
            Remaining embeddings and IDs
        """
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        self.index.index_data(ids_toadd, embeddings_toadd)
        return embeddings, ids
        
    def embed_text(self, text):
        """Convert text to embedding
        
        Args:
            text: Query text
            
        Returns:
            Text embedding vector
        """
        with torch.no_grad():
            encoded_batch = self.tokenizer.batch_encode_plus(
                [text],
                return_tensors="pt",
                max_length=512,
                padding=True,
                truncation=True,
            )
            encoded_batch = {k: v.to(self.device) for k, v in encoded_batch.items()}
            text_embedding = self.text_model(​**​encoded_batch)
            
        return text_embedding.cpu().numpy()
        
    def combine_embeddings(self, text_embedding, video_embedding):
        """Combine text and video embeddings
        
        Args:
            text_embedding: Text embedding
            video_embedding: Video embedding
            
        Returns:
            Combined embedding
        """
        if video_embedding is None:
            return text_embedding
            
        if text_embedding.shape[1] != video_embedding.shape[0]:
            projection = torch.nn.Linear(video_embedding.shape[0], text_embedding.shape[1]).to(self.device)
            
            if self.use_fp16:
                projection = projection.half()
                video_embedding = video_embedding.half()
            else:
                video_embedding = video_embedding.float()
                
            with torch.no_grad():
                video_embedding = projection(video_embedding.to(self.device))
                video_embedding = video_embedding.cpu().numpy().reshape(1, -1)
        else:
            if self.use_fp16:
                video_embedding = video_embedding.half()
            
            video_embedding = video_embedding.cpu().numpy().reshape(1, -1)
            
        combined_embedding = self.text_weight * text_embedding + (1 - self.text_weight) * video_embedding
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding, axis=1, keepdims=True)
        
        return combined_embedding
        
    def search_document(self, query_text, video_path):
        """Search for documents related to query
        
        Args:
            query_text: Query text
            video_path: Path to video file
            
        Returns:
            List of retrieved documents
        """
        print(f"Processing query: {query_text}")
        print(f"Video path: {video_path}")
        
        text_embedding = self.embed_text(query_text)
        
        video_embedding = None
        if video_path and os.path.exists(video_path):
            video_embedding = self.video_processor.process_video(video_path)
        else:
            print(f"Warning: Video file not found or not provided: {video_path}")
            
        combined_embedding = self.combine_embeddings(text_embedding, video_embedding)
        
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(combined_embedding, self.n_docs)
        print(f"Retrieval time: {time.time()-start_time_retrieval:.1f} sec")
        
        docs = [self.passage_id_map[doc_id] for doc_id in top_ids_and_scores[0][0]]
        
        return docs
        
    def process_queries_file(self, queries_file, output_file=None):
        """Process queries file
        
        Args:
            queries_file: Path to queries file (.json or .jsonl)
            output_file: Path to output file
            
        Returns:
            List of retrieval results
        """
        print(f"Loading queries file: {queries_file}")
        if queries_file.endswith(".json"):
            with open(queries_file, "r", encoding="utf-8") as f:
                queries = json.load(f)
        elif queries_file.endswith(".jsonl"):
            queries = []
            with open(queries_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        queries.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format: {queries_file}")
            
        print(f"Loaded {len(queries)} queries")
        
        all_results = []
        for i, item in enumerate(tqdm(queries, desc="Processing queries")):
            query_text = item.get("instruction") or item.get("question", "")
            query_id = item.get("id", f"query_{i}")
            video_path = item.get("video_path", "")
            
            if not query_text:
                print(f"Warning: Query {query_id} has no text content, skipping")
                continue
                
            print(f"\nProcessing query {i+1}/{len(queries)}: {query_id}")
            
            results = self.search_document(query_text, video_path)
            
            result_item = {
                "id": query_id,
                "query": query_text,
                "video_path": video_path,
                "documents": results
            }
            
            all_results.append(result_item)
            
            print(f"Found {len(results)} relevant documents")
            if results:
                print(f"First document ID: {results[0]['id']}")
        
        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output_file}")
        
        return all_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simplified video-text joint retrieval system")
    
    parser.add_argument("--queries", type=str, required=True, help="Path to queries file (.json or .jsonl)")
    parser.add_argument("--passages", type=str, required=True, help="Path to documents file")
    parser.add_argument("--passages_embeddings", type=str, required=True, help="Path to document embeddings (glob pattern)")
    parser.add_argument("--model", type=str, required=True, help="Path to text model")
    
    parser.add_argument("--output", type=str, required=True, help="Path to output file")
    parser.add_argument("--n_docs", type=int, default=8, help="Number of documents to retrieve per query")
    parser.add_argument("--text_weight", type=float, default=0.3, help="Weight for text embedding (0-1)")
    parser.add_argument("--video_fps", type=float, default=1.0, help="Video processing frame rate")
    parser.add_argument("--no_fp16", action="store_true", help="Disable half-precision inference")
    
    args = parser.parse_args()
    
    retriever = SimpleVideoTextRetriever(
        model_path=args.model,
        passages_path=args.passages,
        passages_embeddings_path=args.passages_embeddings,
        video_fps=args.video_fps,
        text_weight=args.text_weight,
        n_docs=args.n_docs,
        use_fp16=not args.no_fp16
    )
    
    retriever.process_queries_file(args.queries, args.output)


if __name__ == "__main__":
    import pickle
    main()