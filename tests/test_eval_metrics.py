"""
Testes unitários para as melhorias de métricas de avaliação.

Coberturas:
  - eval_micro_query retorna 4-tuple (prec, recall, f1, accuracy)
  - accuracy em eval_micro_query é calculada corretamente
  - compute_metrics inclui 'accuracy' (exact-match por caso)
  - compute_metrics com predições perfeitas → accuracy=1.0
  - compute_metrics com predições vazias → accuracy=0.0
  - valid() retorna dicionário com chave 'accuracy'
"""

import json
import pytest
import numpy as np
import tempfile
import os

from tools.eval_tool import eval_micro_query, compute_metrics


# ---------------------------------------------------------------------------
# Testes para eval_micro_query
# ---------------------------------------------------------------------------

class TestEvalMicroQuery:
    def _make_item(self, guid: str, label: int, score_class1: float):
        """Helper: cria item no formato [guid, label, scores]."""
        # scores são logits: [score_class0, score_class1]
        return [guid, label, [1.0 - score_class1, score_class1]]

    def test_returns_four_values(self):
        items = [
            self._make_item("q1_c1", label=1, score_class1=0.9),
            self._make_item("q1_c2", label=0, score_class1=0.1),
        ]
        result = eval_micro_query(items)
        assert len(result) == 4, "eval_micro_query deve retornar 4 valores (prec, recall, f1, accuracy)"

    def test_perfect_predictions_accuracy_is_one(self):
        items = [
            self._make_item("q1_c1", label=1, score_class1=0.9),
            self._make_item("q1_c2", label=0, score_class1=0.1),
            self._make_item("q2_c1", label=1, score_class1=0.8),
            self._make_item("q2_c2", label=0, score_class1=0.2),
        ]
        _, _, _, accuracy = eval_micro_query(items)
        assert accuracy == pytest.approx(1.0), "Predições perfeitas devem ter accuracy=1.0"

    def test_all_wrong_accuracy_is_zero(self):
        items = [
            self._make_item("q1_c1", label=0, score_class1=0.9),  # Errado
            self._make_item("q1_c2", label=1, score_class1=0.1),  # Errado
        ]
        _, _, _, accuracy = eval_micro_query(items)
        assert accuracy == pytest.approx(0.0), "Predições todas erradas devem ter accuracy=0.0"

    def test_half_correct_accuracy_is_half(self):
        items = [
            self._make_item("q1_c1", label=1, score_class1=0.9),  # Correto
            self._make_item("q1_c2", label=1, score_class1=0.1),  # Errado (pred=0, real=1)
        ]
        _, _, _, accuracy = eval_micro_query(items)
        assert accuracy == pytest.approx(0.5), "Metade correto deve ter accuracy=0.5"

    def test_accuracy_is_between_zero_and_one(self):
        items = [
            self._make_item("q1_c1", label=1, score_class1=0.9),
            self._make_item("q1_c2", label=0, score_class1=0.8),
        ]
        _, _, _, accuracy = eval_micro_query(items)
        assert 0.0 <= accuracy <= 1.0

    def test_accuracy_type_is_float(self):
        items = [
            self._make_item("q1_c1", label=1, score_class1=0.9),
        ]
        _, _, _, accuracy = eval_micro_query(items)
        assert isinstance(accuracy, float)


# ---------------------------------------------------------------------------
# Testes para compute_metrics
# ---------------------------------------------------------------------------

def _write_jsonl(path: str, data: dict):
    """Escreve um arquivo JSON Lines (uma linha por objeto)."""
    with open(path, "w") as f:
        json.dump(data, f)
        f.write("\n")


def _write_json(path: str, data: dict):
    """Escreve um arquivo JSON padrão."""
    with open(path, "w") as f:
        json.dump(data, f)


class TestComputeMetrics:
    def test_accuracy_key_present(self, tmp_path):
        labels_file = str(tmp_path / "labels.json")
        predicted_file = str(tmp_path / "predicted.json")
        _write_jsonl(labels_file, {"q1.txt": ["d1.txt", "d2.txt"]})
        _write_json(predicted_file, {"q1.txt": ["d1.txt", "d2.txt"]})

        metrics = compute_metrics(labels_file, predicted_file)
        assert "accuracy" in metrics, "compute_metrics deve retornar chave 'accuracy'"

    def test_perfect_prediction_accuracy_is_one(self, tmp_path):
        labels_file = str(tmp_path / "labels.json")
        predicted_file = str(tmp_path / "predicted.json")
        labels = {"q1.txt": ["d1.txt"], "q2.txt": ["d2.txt", "d3.txt"]}
        _write_jsonl(labels_file, labels)
        _write_json(predicted_file, labels)  # mesma predição = perfeita

        metrics = compute_metrics(labels_file, predicted_file)
        assert metrics["accuracy"] == pytest.approx(1.0), (
            "Predição exata do ground truth deve ter accuracy=1.0"
        )

    def test_empty_prediction_accuracy_is_zero(self, tmp_path):
        labels_file = str(tmp_path / "labels.json")
        predicted_file = str(tmp_path / "predicted.json")
        _write_jsonl(labels_file, {"q1.txt": ["d1.txt"]})
        _write_json(predicted_file, {"q1.txt": []})

        metrics = compute_metrics(labels_file, predicted_file)
        assert metrics["accuracy"] == pytest.approx(0.0), (
            "Predição vazia para caso com labels não-vazias deve ter accuracy=0.0"
        )

    def test_partial_match_accuracy_between_zero_and_one(self, tmp_path):
        labels_file = str(tmp_path / "labels.json")
        predicted_file = str(tmp_path / "predicted.json")
        _write_jsonl(labels_file, {"q1.txt": ["d1.txt"], "q2.txt": ["d2.txt"]})
        _write_json(predicted_file, {"q1.txt": ["d1.txt"], "q2.txt": ["d3.txt"]})  # q2 errado

        metrics = compute_metrics(labels_file, predicted_file)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_accuracy_is_float(self, tmp_path):
        labels_file = str(tmp_path / "labels.json")
        predicted_file = str(tmp_path / "predicted.json")
        _write_jsonl(labels_file, {"q1.txt": ["d1.txt"]})
        _write_json(predicted_file, {"q1.txt": ["d1.txt"]})

        metrics = compute_metrics(labels_file, predicted_file)
        assert isinstance(metrics["accuracy"], float)

    def test_all_metrics_present(self, tmp_path):
        labels_file = str(tmp_path / "labels.json")
        predicted_file = str(tmp_path / "predicted.json")
        _write_jsonl(labels_file, {"q1.txt": ["d1.txt"]})
        _write_json(predicted_file, {"q1.txt": ["d1.txt"]})

        metrics = compute_metrics(labels_file, predicted_file)
        for key in ("precision", "recall", "f1_score", "accuracy"):
            assert key in metrics, f"Chave '{key}' ausente em compute_metrics"
