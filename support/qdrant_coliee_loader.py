"""
Script to load case pair data (q_paras and c_paras) from JSON file into Qdrant.
Each paragraph is stored as a Point with embeddings generated using all-MiniLM-L6-v2.
"""
import json
import os
from typing import Dict, List, Tuple, Iterator
import hashlib
import logging

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ray
import torch

logger = logging.getLogger(__name__)

def load_embedding_model() -> SentenceTransformer:
    """Load the all-MiniLM-L6-v2 sentence transformer model."""
    # return SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    return SentenceTransformer('all-MiniLM-L6-v2', device='cuda')


def generate_point_id(document_name: str, idx: int, text: str) -> int:
    """Generate a unique point ID based on document name, index, and text."""
    content = f"{document_name}_{idx}_{text[:100]}"  # Use first 100 chars to avoid collisions
    hash_object = hashlib.md5(content.encode())
    return int(hash_object.hexdigest(), 16) & 0xFFFFFFFF


def process_paragraphs(
    document_name: str,
    paragraphs: List[str],
    model: SentenceTransformer
) -> List[PointStruct]:
    """
    Process a list of paragraphs into PointStruct objects.
    
    Args:
        document_name: Name of the document
        paragraphs: List of paragraph texts
        model: Sentence transformer model for embeddings
        
    Returns:
        List of PointStruct objects
    """
    points = []
    
    # Generate embeddings for all paragraphs at once for efficiency
    embeddings = model.encode(paragraphs, convert_to_tensor=False, show_progress_bar=False)
    
    for idx, (text, embedding) in enumerate(zip(paragraphs, embeddings)):
        if not text.strip():  # Skip empty paragraphs
            continue
            
        payload = {
            "document_name": document_name,
            "idx": idx,
            "text": text.strip()
        }
        
        point_id = generate_point_id(document_name, idx, text)
        
        # Convert numpy array to list for JSON serialization
        vector = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        
        points.append(PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        ))
    
    return points

def point_exists(
    client: QdrantClient,
    collection_name: str,
    document_name: str
) -> bool:
    """
    Check if any point with the given document name exists in the Qdrant collection.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the Qdrant collection
        document_name: Document name to check
        
    Returns:
        True if point exists, False otherwise
    """
    logger = logging.getLogger(__name__)
    try:
        scroll_result, next_page_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="document_name", 
                                          match=models.MatchValue(value=document_name)),
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
            offset=None
        )
        logger.info(f"Checked existence of {document_name} in Qdrant.")
        logger.info(f"Scroll result: {scroll_result}")
        return bool(scroll_result)
    except Exception as e:
        logger.error(f"Error checking existence of {document_name} in Qdrant: {e}")
        return False

@ray.remote(num_gpus=0.25)
def process_case_pair_batch(
    batch: List[Dict],
    collection_name: str,
    qdrant_host: str = "qdrant",
    qdrant_port: int = 6333,
    qdrant_grcport: int = 6334
) -> int:
    """
    Process a batch of case pairs in parallel using Ray.
    
    Args:
        batch: List of case pair dictionaries
        collection_name: Name of the Qdrant collection
        qdrant_host: Qdrant host
        qdrant_port: Qdrant port
        
    Returns:
        Number of points processed
    """
    client = QdrantClient(host=qdrant_host, port=qdrant_port, 
                    grpc_port=qdrant_grcport, prefer_grpc=True)
    model = load_embedding_model()
    
    all_points = []
    
    for case_pair in tqdm(batch, desc="Processing case pairs", leave=False):
        guid = case_pair.get("guid", "")
        q_paras = case_pair.get("q_paras", [])
        c_paras = case_pair.get("c_paras", [])
        
        if not guid or "_" not in guid:
            logger.warning(f"Invalid guid format: {guid}")
            continue
            
        # Extract document names from guid
        parts = guid.split("_", 1)  # Split only on first underscore
        q_doc_name = parts[0] + ".txt"
        c_doc_name = parts[1] + ".txt"
        # Check if q_doc_name already exists in Qdrant collection
        # Process q_paras
        if q_paras and not point_exists(client, collection_name, q_doc_name):
            q_points = process_paragraphs(q_doc_name, q_paras, model)
            all_points.extend(q_points)
        
        # Process c_paras
        if c_paras and not point_exists(client, collection_name, c_doc_name):
            c_points = process_paragraphs(c_doc_name, c_paras, model)
            all_points.extend(c_points)
    
    # Insert points into Qdrant
    if all_points:
        client.upsert(collection_name=collection_name, points=all_points)
    
    return len(all_points)


def stream_case_pairs(json_path: str, batch_size: int = 10) -> Iterator[List[Dict]]:
    """
    Stream case pairs from JSON file in batches.
    
    Args:
        json_path: Path to the JSON file
        batch_size: Number of case pairs per batch
        
    Yields:
        Batches of case pair dictionaries
    """
    current_batch = []
    
    with open(json_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            try:
                case_pair = json.loads(line)
                current_batch.append(case_pair)
                
                if len(current_batch) >= batch_size:
                    yield current_batch
                    current_batch = []
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON line: {e}")
                continue
    
    # Yield remaining items
    if current_batch:
        yield current_batch


def create_qdrant_collection(
    collection_name: str,
    vector_size: int = 384,  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    qdrant_host: str = "qdrant",
    qdrant_port: int = 6333,
    drop_if_exists: bool = False
) -> None:
    """
    Create a Qdrant collection if it doesn't exist.
    
    Args:
        collection_name: Name of the collection
        vector_size: Size of the embedding vectors
        qdrant_host: Qdrant host
        qdrant_port: Qdrant port
    """
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    if drop_if_exists and client.collection_exists(collection_name=collection_name):
        logger.info(f"Dropping existing collection: {collection_name}")
        client.delete_collection(collection_name=collection_name)
    
    if not client.collection_exists(collection_name=collection_name):
        logger.info(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    else:
        logger.info(f"Collection already exists: {collection_name}")


def load_case_pairs_to_qdrant(
    json_path: str,
    collection_name: str = "case_pairs",
    batch_size: int = 100,
    qdrant_host: str = "qdrant",
    qdrant_port: int = 6333,
    qdrant_grpc_port: int = 6334
) -> None:
    """
    Load case pairs from JSON file into Qdrant collection.
    
    Args:
        json_path: Path to the JSON file
        collection_name: Name of the Qdrant collection
        batch_size: Number of case pairs to process per batch
        qdrant_host: Qdrant host
        qdrant_port: Qdrant port
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    # Create collection
    create_qdrant_collection(collection_name, qdrant_host=qdrant_host,
                             qdrant_port=qdrant_port, drop_if_exists=True)
    
    logger.info(f"Loading case pairs from: {json_path}")
    # Initialize Ray if not already initialized
    # if not ray.is_initialized():
    ray.init(num_cpus=4, ignore_reinit_error=True, runtime_env={"working_dir": ".", 
                                                                "excludes": ["/data/", "/output/"]})
    
    # Process batches in parallel
    futures = []
    total_batches = 0
    
    for batch in tqdm(stream_case_pairs(json_path, batch_size), desc="Creating batches"):
        futures.append(
            process_case_pair_batch.remote(
                batch, collection_name, qdrant_host, qdrant_grpc_port
            )
        )
        total_batches += 1
    
    # Collect results and show progress
    total_points = 0
    for points_count in tqdm(
        ray.get(futures), 
        total=total_batches, 
        desc="Processing case pair batches"
    ):
        total_points += points_count
    
    logger.info(f"Successfully loaded {total_points} points to collection '{collection_name}'")


def main(
    json_path: str = "data/sample_test_paragraphs_processed_data.json",
    collection_name: str = "case_pairs", 
    batch_size: int = 10
) -> None:
    """
    Main function to load case pairs into Qdrant.
    
    Args:
        json_path: Path to the JSON file containing case pairs
        collection_name: Name of the Qdrant collection
    """
    try:
        load_case_pairs_to_qdrant(json_path, collection_name, batch_size=batch_size)
    except Exception as e:
        logger.error(f"Error loading case pairs: {e}")
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    import argparse
    print("CUDA AVAILABLE:", torch.cuda.is_available())
    parser = argparse.ArgumentParser(description="Load case pairs into Qdrant collection")
    parser.add_argument(
        "json_path", 
        type=str, 
        help="Path to the JSON file containing case pairs"
    )
    parser.add_argument(
        "--collection-name", 
        type=str, 
        default="case_pairs",
        help="Name of the Qdrant collection (default: case_pairs)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of case pairs per batch (default: 100)"
    )
    
    args = parser.parse_args()
    main(args.json_path, args.collection_name, args.batch_size)
