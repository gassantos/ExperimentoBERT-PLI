import argparse
import numpy as np
import hdbscan
from qdrant_client import QdrantClient, models
from typing import List, Dict, Tuple, Any
import scipy
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm
import joblib

# ideia https://milvus.io/docs/hdbscan_clustering_with_milvus.md

# --- Configuração ---
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
QDRANT_GRPC_PORT = 6334
COLLECTION_NAME = "coliee-test"
K_NEIGHBORS = 100  # Número de vizinhos a serem considerados para cada ponto
BATCH_SIZE = 100

# Parâmetros do HDBSCAN
HDBSCAN_PARAMS = {
    "min_cluster_size": 10,
    "min_samples": 5,
    "metric": "precomputed",
    "cluster_selection_method": "eom",
}


def run_hdbscan_clustering(distance_matrix: lil_matrix) -> hdbscan.HDBSCAN:
    """
    Executa o clustering HDBSCAN em uma matriz de distâncias pré-computada (esparsa ou densa).
    """
    print("Executando o clustering HDBSCAN...")
    max_dist = (distance_matrix.max() - distance_matrix.min()) * 1000
    hdbscan_params = HDBSCAN_PARAMS.copy()
    hdbscan_params["max_dist"] = max_dist
    clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    connected_graphs = attach_fully_connected_node(
        distance_matrix.tocsr(), dist_fullyConnectedNode=max_dist
    )
    clusterer.fit(connected_graphs)
    print("Clustering concluído.")
    return clusterer


def process_and_update_results(
    client: QdrantClient,
    collection_name: str,
    point_ids: List[Any],
    payloads: List[Any],
    embeddings: List[Any],
    clusterer: hdbscan.HDBSCAN,
):
    """
    Processa os resultados do clustering e atualiza os payloads no Qdrant.
    """
    num_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    num_noise = np.sum(clusterer.labels_ == -1)
    print(
        f"Resultados: {num_clusters} clusters encontrados, com {num_noise} pontos de ruído."
    )
    print("Atualizando payloads no Qdrant com os labels dos clusters...")
    payload_updates = []
    for id, c, p in zip(point_ids, clusterer.labels_.tolist(), payloads):
        p["cluster"] = c
        payload_updates.append({"id": id, "payload": p})

    # Update points
    if payload_updates:
        payload_operations = [
            models.SetPayloadOperation(
                set_payload=models.SetPayload(
                    payload=payload["payload"],
                    points=[payload["id"]],
                )
            )
            for payload in payload_updates
        ]

        for i in tqdm(
            range(0, len(payload_operations), BATCH_SIZE),
            desc="Processing batches"
        ):
            batch_payloads = payload_operations[i : i + BATCH_SIZE]
            client.batch_update_points(
                collection_name=collection_name, update_operations=batch_payloads
            )

    print("Payloads atualizados com sucesso.")


def build_distance_matrix(
    client: QdrantClient, collection_name: str, k: int
) -> np.ndarray:

    ids = []
    dist = {}
    embeddings = []
    payloads = []
    next_offset = None

    progress = tqdm(desc="Building distance matrix", unit=" batches")

    while True:
        batch, next_offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=next_offset,
            with_payload=True,
            with_vectors=True,
        )

        batch_ids = [point.id for point in batch]
        ids.extend(batch_ids)
        query_vectors = [point.vector for point in batch]
        embeddings.extend(query_vectors)
        payloads.extend([point.payload for point in batch])

        search_queries = [
            models.QueryRequest(query=vector, limit=k + 1) for vector in query_vectors
        ]
        results = client.query_batch_points(
            collection_name=collection_name,
            requests=search_queries,
        )
        for i, batch_id in enumerate(batch_ids):
            dist[batch_id] = []
            for result in results[i].points:
                dist[batch_id].append((result.id, 1 - result.score))

        progress.update()

        if next_offset is None:
            break  # No more pages

    progress.close()

    ids2index = {}

    for id in dist:
        ids2index[id] = len(ids2index)

    dist_metric = lil_matrix((len(ids), len(ids)), dtype=np.float64)

    for id in dist:
        for result in dist[id]:
            dist_metric[ids2index[id], ids2index[result[0]]] = result[1]

    return ids, payloads, embeddings, dist_metric.tocsr()


# https://github.com/scikit-learn-contrib/hdbscan/issues/82
def attach_fully_connected_node(d, dist_fullyConnectedNode=None):
    """
    This function takes in a sparse graph (csr_matrix) that has more than
     one component (multiple unconnected subgraphs) and appends another
     node to the graph that is weakly connected to all other nodes.
    RH 2022

    Args:
        d (scipy.sparse.csr_matrix):
            Sparse graph with multiple components.
            See scipy.sparse.csgraph.connected_components
        dist_fullyConnectedNode (float):
            Value to use for the connection strengh to all other nodes.
            Value will be appended as elements in a new row and column at
             the ends of the 'd' matrix.

     Returns:
         d2 (scipy.sparse.csr_matrix):
             Sparse graph with only one component.
    """
    if dist_fullyConnectedNode is None:
        dist_fullyConnectedNode = (d.max() - d.min()) * 1000

    d2 = d.copy()
    d2 = scipy.sparse.vstack((d2, np.ones((1, d2.shape[1])) * dist_fullyConnectedNode))
    d2 = scipy.sparse.hstack((d2, np.ones((d2.shape[0], 1)) * dist_fullyConnectedNode))

    return d2.tocsr()


def load_serialized_clusterer(path: str, client: QdrantClient):
    point_ids = []
    payloads = []
    embeddings = []
    next_offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=next_offset,
            with_payload=True,
            with_vectors=True,
        )

        batch_ids = [point.id for point in batch]
        point_ids.extend(batch_ids)
        payloads.extend([point.payload for point in batch])
        query_vectors = [point.vector for point in batch]
        embeddings.extend(query_vectors)
        if next_offset is None:
            break  # No more pages
    # Carregar o clusterer salvo
    clusterer = joblib.load(path)
    return point_ids, payloads, embeddings, clusterer


def main(collection_name: str, output_file: str):
    """
    Orquestra o pipeline completo de clustering escalável.
    """
    client = QdrantClient(
        host=QDRANT_HOST, port=QDRANT_PORT, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True
    )
    point_ids, payloads, embeddings, distance_matrix = build_distance_matrix(
        client, collection_name, K_NEIGHBORS
    )

    # # 3. Executar clustering
    clusterer = run_hdbscan_clustering(distance_matrix)
    joblib.dump(clusterer, output_file)

    # point_ids, payloads, embeddings, clusterer = load_serialized_clusterer(
    #     output_file, client
    # )
    # 4. Processar e utilizar os resultados
    process_and_update_results(
        client, collection_name, point_ids, payloads, embeddings, clusterer
    )


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
        "--output-file",
        type=str,
        # required=True,
        default="output/checkpoints/hdbscan.joblib",
        help="Caminho para o arquivo de saída (padrão: output/checkpoints/hdbscan.joblib)",
    )

    args = parser.parse_args()
    collection_name = args.collection
    output_file = args.output_file

    main(collection_name, output_file)
