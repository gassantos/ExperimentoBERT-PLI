import argparse
import random
import logging
import json
from typing import Any, Dict, List
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

random.seed(42)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuração ---
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
QDRANT_GRPC_PORT = 6334
COLLECTION_NAME = "coliee-test"
K_NEIGHBORS = 100  # Número de vizinhos a serem considerados para cada ponto
BATCH_SIZE = 100
PREFER_GRPC = True  # Preferir gRPC para comunicação com Qdrant


def get_relevant_clustered_segments(
    client: QdrantClient,
    collection_name: str,
    document_name: str,
    batch_size: int = 100,
) -> List[Dict]:
    """
    Recupera todos os segmentos relevantes para um documento específico.

    Args:
        document_name: Nome do documento para recuperar segmentos
        batch_size: Tamanho do lote para operações de scroll

    Returns:
        Lista de segmentos relevantes
    """
    logger.debug(
        f"Recuperando segmentos relevantes para o documento: {document_name}.txt"
    )

    # Criar filtro para buscar segmentos relevantes do documento específico
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="document_name", match=MatchValue(value=f"{document_name}.txt")
            ),
            FieldCondition(key="relevance", match=MatchValue(value=True)),
            FieldCondition(
                key="cluster",
                range=models.Range(
                    gte=0,
                ),
            ),
        ]
    )

    segments = []
    offset = None
    progress = tqdm(desc="Retrieving segments", unit=" batches")

    # Usar scroll para buscar todos os resultados em lotes
    while True:
        batch_results, next_page_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_condition,
            limit=batch_size,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )

        segments.extend(batch_results)

        if not batch_results:
            break

        if next_page_offset is None:
            break

        offset = next_page_offset
        progress.update()
    progress.close()
    # Ordenar segmentos por posição no documento
    segments_sorted = sorted(segments, key=lambda x: x.payload.get("position", 0))

    logger.debug(f"Recuperados {len(segments_sorted)} segmentos para {document_name}")
    return segments_sorted


def main(
    collection_name: str, json_path: str, output_file: str
) -> List[Dict[str, Any]]:
    """
    Carrega um arquivo JSONL e retorna uma lista de dicionários.
    """
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        grpc_port=QDRANT_GRPC_PORT,
        prefer_grpc=PREFER_GRPC,
    )

    with open(json_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            q_name, c_name = sample["guid"].split("_")
            c_segments = get_relevant_clustered_segments(
                client, collection_name, c_name
            )
            c_texts = [segment.payload.get("text", "") for segment in c_segments]
            q_segments = get_relevant_clustered_segments(
                client, collection_name, q_name
            )
            q_texts = [segment.payload.get("text", "") for segment in q_segments]
            sample["q_paras"] = q_texts
            sample["c_paras"] = c_texts

            with open(output_file, "w") as f:
                f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrai segmentos relevantes do Qdrant e gera um arquivo JSON estruturado"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        # required=True,
        help="Nome da coleção no Qdrant (padrão: coliee-test)",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        # required=True,
        default="data/test_paragraphs_processed_data.json",
        help="Caminho para o arquivo JSON de entrada (padrão: data/test_paragraphs_processed_data.json)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        # required=True,
        default="data/test_relevant_clustered_segments.json",
        help="Caminho para o arquivo de saída (padrão: data/test_relevant_clustered_segments.json)",
    )

    args = parser.parse_args()
    collection_name = args.collection
    json_path = args.json_path
    output_file = args.output_file
    data = main(collection_name, json_path, output_file)
