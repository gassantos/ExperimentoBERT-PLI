import logging
import torch
import numpy as np
from collections import defaultdict
from timeit import default_timer as timer

logger = logging.getLogger(__name__)


def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60
    return '%2d:%02d' % (minute, second)


def output_value(epoch, mode, step, time, loss, info, end, config):
    try:
        delimiter = config.get("output", "delimiter")
    except Exception:
        delimiter = " "
    s = ""
    s = s + str(epoch) + " "
    while len(s) < 7:
        s += " "
    s = s + str(mode) + " "
    while len(s) < 14:
        s += " "
    s = s + str(step) + " "
    while len(s) < 25:
        s += " "
    s += str(time)
    while len(s) < 40:
        s += " "
    s += str(loss)
    while len(s) < 48:
        s += " "
    s += str(info)
    s = s.replace(" ", delimiter)
    if end is not None:
        print(s, end=end)
    else:
        print(s)


def eval_micro_query(_result_list):
    label_dict = defaultdict(lambda: [])
    pred_dict = defaultdict(lambda: defaultdict(lambda: 0))
    pair_correct = 0
    total_pairs = len(_result_list)
    for item in _result_list:
        guid = item[0]
        label = int(item[1])
        pred = np.argmax(item[2])
        qid, cid = guid.split('_')
        # if label > 0:
        label_dict[qid].append(cid)
        pred_dict[qid][cid] = pred
        if pred == label:
            pair_correct += 1
    assert (len(pred_dict) == len(label_dict))

    correct = 0
    label = 0
    predict = 0
    for qid in label_dict:
        label += len(label_dict[qid])
    for qid in pred_dict:
        for cid in pred_dict[qid]:
            if pred_dict[qid][cid] == 1:
                predict += 1
                if cid in label_dict[qid]:
                    correct += 1
    if correct == 0:
        micro_prec_query = 0
        micro_recall_query = 0
    else:
        micro_prec_query = float(correct) / predict
        micro_recall_query = float(correct) / label
    if micro_prec_query > 0 or micro_recall_query > 0:
        micro_f1_query = (2 * micro_prec_query * micro_recall_query) / (micro_prec_query + micro_recall_query)
    else:
        micro_f1_query = 0
    accuracy = pair_correct / total_pairs if total_pairs > 0 else 0.0
    return micro_prec_query, micro_recall_query, micro_f1_query, accuracy


def valid(model, dataset, epoch, writer, config, gpu_list, output_function, mode="valid"):
    model.eval()

    acc_result = None
    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = ""

    output_time = config.getint("output", "output_time")
    step = -1
    result = []

    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = data[key].cuda()

        results = model(data, config, gpu_list, acc_result, "valid")
        loss, acc_result, output = results["loss"], results["acc_result"], results["output"]
        total_loss += loss.detach().item()
        result = result + output
        cnt += 1

        if step % output_time == 0:
            delta_t = timer() - start_time

            output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    delta_t = timer() - start_time
    output_info = output_function(acc_result, config)
    output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                 "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

    writer.add_scalar(config.get("output", "model_name") + "_eval_epoch", float(total_loss) / (step + 1),
                      epoch)

    # eval results based on query micro F1
    micro_prec_query, micro_recall_query, micro_f1_query, accuracy = eval_micro_query(result)
    loss_tmp = total_loss / (step + 1)
    print('valid set: micro_prec_query=%.4f, micro_recall_query=%.4f, micro_f1_query=%.4f, accuracy=%.4f' %
          (micro_prec_query, micro_recall_query, micro_f1_query, accuracy))
    model.train()
    return {'precision': micro_prec_query, 'recall': micro_recall_query, 'f1': micro_f1_query,
            'accuracy': accuracy, 'loss': loss_tmp}


# ---------------------------------------------------------------------------
# Avaliação de resultados (parse + métricas)
# ---------------------------------------------------------------------------

def parse_gru_results(input_file: str, output_file: str) -> None:
    """
    Converte saída bruta do GRU/LSTM (lista de pares ``[case_pair, scores]``)
    para o formato task1: ``{query_file: [doc_file, ...]}``.

    Args:
        input_file: Arquivo JSON com resultados brutos do GRU/LSTM
        output_file: Arquivo JSON de saída no formato task1
    """
    from pathlib import Path as _Path
    import json as _json
    from utils.paths import PathManager

    output_path = _Path(output_file)
    PathManager.ensure_dir(output_path.parent)

    with open(input_file, "r") as f:
        data = _json.load(f)

    result: dict = {}

    for entry in data:
        if not entry or len(entry) != 2:
            continue
        case_pair, scores = entry[0], entry[1]
        if not scores or len(scores) != 2:
            continue

        parts = case_pair.split("_")
        if len(parts) != 2:
            continue

        query, doc = parts[0], parts[1]
        score_irrelevant, score_relevant = scores[0], scores[1]

        if score_relevant > score_irrelevant:
            query_file = f"{query}.txt"
            if query_file not in result:
                result[query_file] = []
            result[query_file].append(f"{doc}.txt")

    for key in result:
        result[key] = sorted(set(result[key]))

    PathManager.ensure_dir(output_path.parent)
    with open(output_file, "w") as f:
        _json.dump(result, f, indent=2, sort_keys=True)

    print(f"Parseados {len(data)} entradas → {len(result)} mapeamentos de casos")
    print(f"Saída: {output_file}")


def compute_metrics(labels_file: str, predicted_file: str, k_values: list = None) -> dict:
    """
    Calcula precisão, recall e F1 (geral e @k) comparando predições com ground truth.

    Args:
        labels_file: Arquivo ground truth (JSON Lines, um objeto por linha)
        predicted_file: Arquivo de predições (JSON padrão)
        k_values: Lista de valores de k para métricas @k (padrão: [1, 3, 5, 10])

    Returns:
        Dicionário com todas as métricas calculadas
    """
    import json as _json

    if k_values is None:
        k_values = [1, 3, 5, 10]

    # Carrega ground truth (JSON Lines)
    labels: dict = {}
    with open(labels_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = _json.loads(line)
                labels.update(obj)

    # Carrega predições (JSON padrão)
    with open(predicted_file, "r") as f:
        predicted: dict = _json.load(f)

    # Normaliza chaves (remove .txt)
    def _norm(key: str) -> str:
        return key[:-4] if key.endswith(".txt") else key

    labels = {_norm(k): [_norm(v) for v in vs] for k, vs in labels.items()}
    predicted = {_norm(k): [_norm(v) for v in vs] for k, vs in predicted.items()}

    total_tp = total_pred = total_actual = 0
    k_metrics = {k: {"tp": 0, "predicted": 0, "actual": 0} for k in k_values}

    for case in set(labels) | set(predicted):
        true_set = set(labels.get(case, []))
        pred_list = predicted.get(case, [])
        pred_set = set(pred_list)

        tp = len(true_set & pred_set)
        total_tp += tp
        total_pred += len(pred_set)
        total_actual += len(true_set)

        for k in k_values:
            pred_at_k = set(pred_list[:k])
            k_metrics[k]["tp"] += len(true_set & pred_at_k)
            k_metrics[k]["predicted"] += len(pred_at_k)
            k_metrics[k]["actual"] += len(true_set)

    precision = total_tp / total_pred if total_pred else 0.0
    recall = total_tp / total_actual if total_actual else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    all_cases = set(labels) | set(predicted)
    exact_match = sum(
        1 for case in all_cases
        if set(labels.get(case, [])) == set(predicted.get(case, []))
    )
    accuracy = exact_match / len(all_cases) if all_cases else 0.0

    results = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "total_cases": len(all_cases),
        "total_true_positives": total_tp,
        "total_predicted": total_pred,
        "total_actual": total_actual,
    }

    for k in k_values:
        ktp = k_metrics[k]["tp"]
        kp = k_metrics[k]["predicted"]
        ka = k_metrics[k]["actual"]
        pk = ktp / kp if kp else 0.0
        rk = ktp / ka if ka else 0.0
        f1k = (2 * pk * rk) / (pk + rk) if (pk + rk) else 0.0
        results[f"precision@{k}"] = pk
        results[f"recall@{k}"] = rk
        results[f"f1_score@{k}"] = f1k

    return results


def evaluate_predictions(
    labels_file: str,
    predicted_file: str,
    output_file: str = "",
) -> dict:
    """
    Avalia predições contra o ground truth e imprime um relatório.

    Args:
        labels_file: Arquivo ground truth (JSON Lines)
        predicted_file: Arquivo de predições (JSON padrão)
        output_file: Caminho opcional para salvar métricas como JSON

    Returns:
        Dicionário com todas as métricas calculadas
    """
    from pathlib import Path as _Path
    import json as _json

    print("Avaliando predições...")
    print(f"  Ground truth : {labels_file}")
    print(f"  Predições    : {predicted_file}")
    print("-" * 50)

    metrics = compute_metrics(labels_file, predicted_file)

    print("Métricas gerais:")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print()

    print("Métricas @k:")
    for k in [1, 3, 5, 10]:
        if f"precision@{k}" in metrics:
            print(
                f"  @{k:2d}  P={metrics[f'precision@{k}']:.4f}  "
                f"R={metrics[f'recall@{k}']:.4f}  "
                f"F1={metrics[f'f1_score@{k}']:.4f}"
            )
    print()

    print("Sumário:")
    print(f"  Casos totais      : {metrics['total_cases']}")
    print(f"  True positives    : {metrics['total_true_positives']}")
    print(f"  Total preditos    : {metrics['total_predicted']}")
    print(f"  Total reais       : {metrics['total_actual']}")

    if output_file:
        _Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            _json.dump(metrics, f, indent=2, sort_keys=True)
        print(f"\nResultados salvos em: {output_file}")

    return metrics


def convert_test_results_to_task1(test_results_file: str) -> dict:
    """
    Converte a saída de ``scripts/test.py`` para o formato task1
    ``{"query.txt": ["doc1.txt", ...]}``, com suporte aos dois formatos
    possíveis de GUID:

    * **Flat** (``"query_doc"``): item de parágrafo já pré-processado.
      Classificado como relevante se ``scores[1] > scores[0]``.
    * **Expandido** (``"query_doc___qi_ci"``): pares gerados internamente pelo
      ``BertPairTextFormatter`` a partir de ``q_paras``/``c_paras``.
      Usa **max pooling** sobre ``scores[1]`` para agregar os pares de
      parágrafo de volta ao par caso-documento original.

    Args:
        test_results_file: Arquivo JSON Lines gerado por ``scripts/test.py``
            (cada linha: ``{"id_": "<guid>", "res": [s_irrel, s_rel]}``).

    Returns:
        Dicionário no formato task1 pronto para :func:`compute_metrics`.
    """
    import json as _json

    _SEP = "___"  # mesmo separador de BertPairTextFormatter

    # guid_base → (best_score_irrel, best_score_rel)
    best_scores: dict = {}

    with open(test_results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = _json.loads(line)
            guid = item.get("id_", "")
            scores = item.get("res", [])
            if not guid or len(scores) != 2:
                continue

            # Strip de sufixo de parágrafo, se existir
            base_guid = guid.split(_SEP)[0]

            # Max pooling: mantém o par com maior score de relevância
            prev = best_scores.get(base_guid)
            if prev is None or scores[1] > prev[1]:
                best_scores[base_guid] = (scores[0], scores[1])

    result: dict = {}
    for base_guid, (s_irrel, s_rel) in best_scores.items():
        parts = base_guid.split("_")
        if len(parts) != 2:
            continue
        query, doc = parts[0], parts[1]
        if s_rel > s_irrel:
            query_file = f"{query}.txt"
            if query_file not in result:
                result[query_file] = []
            result[query_file].append(f"{doc}.txt")

    for key in result:
        result[key] = sorted(set(result[key]))
    return result
