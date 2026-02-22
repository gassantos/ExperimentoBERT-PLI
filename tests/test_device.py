"""
Testes unitários para utils/device.py.

Coberturas:
  - get_device(): CPU forçada, CUDA, MPS, fallback CPU, torch indisponível
  - get_device_info(): chaves obrigatórias, campos CUDA, torch indisponível
  - set_device_optimization(): CUDA (TF32/benchmark), MPS, CPU, torch/device None
  - get_torch_device(): retorno em ambiente GPU, CPU e torch indisponível

Estratégia de mock:
  - ``utils.device.torch`` é substituído por ``None`` para simular ausência do PyTorch.
  - ``torch.cuda.is_available`` é patchado para controlar o ramo CUDA/CPU.
  - ``torch.backends.mps`` e ``platform.system`` são patchados para o ramo MPS.
  - ``utils.device.xm`` e ``utils.device._XLA_AVAILABLE`` cobrem o ramo TPU.
"""
from __future__ import annotations

import platform
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

import utils.device as dev


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cpu_device() -> torch.device:
    return torch.device("cpu")


def _cuda_device() -> torch.device:
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# get_device()
# ---------------------------------------------------------------------------

class TestGetDevice:
    """Testes para get_device()."""

    def test_returns_torch_device(self):
        """Retorno é sempre um torch.device quando PyTorch está disponível."""
        device = dev.get_device()
        assert isinstance(device, torch.device), (
            "get_device() deve retornar torch.device quando PyTorch disponível"
        )

    def test_prefer_cpu_returns_cpu(self):
        """prefer_cpu=True sempre retorna CPU, mesmo com GPU disponível."""
        device = dev.get_device(prefer_cpu=True)
        assert device is not None
        assert device.type == "cpu", (
            "get_device(prefer_cpu=True) deve retornar device de tipo 'cpu'"
        )

    def test_returns_none_when_torch_unavailable(self):
        """Retorna None quando torch não está disponível."""
        with patch.object(dev, "torch", None):
            result = dev.get_device()
        assert result is None

    def test_cuda_branch_when_cuda_available(self):
        """Retorna device 'cuda' quando torch.cuda.is_available() é True."""
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1e9
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_name", return_value="Tesla T4"), \
             patch("torch.cuda.get_device_properties", return_value=mock_props):
            device = dev.get_device()
        assert device is not None
        assert device.type == "cuda"

    def test_cpu_fallback_when_no_gpu(self):
        """Retorna device 'cpu' quando CUDA e MPS não estão disponíveis."""
        with patch("torch.cuda.is_available", return_value=False), \
             patch("platform.system", return_value="Linux"):
            device = dev.get_device()
        assert device is not None
        assert device.type == "cpu"

    def test_mps_branch_on_darwin(self):
        """Retorna device 'mps' em macOS com MPS disponível."""
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = True
        mock_backends = MagicMock()
        mock_backends.mps = mock_mps

        with patch("torch.cuda.is_available", return_value=False), \
             patch("platform.system", return_value="Darwin"), \
             patch("torch.backends", mock_backends):
            device = dev.get_device()
        assert device is not None
        assert device.type == "mps"

    def test_cpu_fallback_on_darwin_without_mps(self):
        """Retorna 'cpu' em macOS quando MPS não está disponível."""
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = False
        mock_backends = MagicMock()
        mock_backends.mps = mock_mps

        with patch("torch.cuda.is_available", return_value=False), \
             patch("platform.system", return_value="Darwin"), \
             patch("torch.backends", mock_backends):
            device = dev.get_device()
        assert device is not None
        assert device.type == "cpu"


# ---------------------------------------------------------------------------
# get_device_info()
# ---------------------------------------------------------------------------

class TestGetDeviceInfo:
    """Testes para get_device_info()."""

    # Chaves que devem estar presentes em qualquer ambiente
    _BASE_KEYS = {"device_type", "platform", "platform_version",
                  "python_version", "pytorch_version"}

    def test_returns_dict(self):
        """Retorno é um dicionário."""
        assert isinstance(dev.get_device_info(), dict)

    def test_base_keys_present(self):
        """Chaves básicas sempre presentes quando PyTorch disponível."""
        info = dev.get_device_info()
        assert self._BASE_KEYS.issubset(info.keys()), (
            f"Chaves ausentes: {self._BASE_KEYS - info.keys()}"
        )

    def test_pytorch_version_is_string(self):
        """pytorch_version é uma string não vazia."""
        info = dev.get_device_info()
        assert isinstance(info["pytorch_version"], str)
        assert len(info["pytorch_version"]) > 0

    def test_platform_matches_current(self):
        """Campo 'platform' corresponde ao sistema atual."""
        info = dev.get_device_info()
        assert info["platform"] == platform.system()

    def test_unavailable_when_torch_none(self):
        """Retorna dict com 'unavailable' e chave 'error' quando torch=None."""
        with patch.object(dev, "torch", None):
            info = dev.get_device_info()
        assert info["device_type"] == "unavailable"
        assert "error" in info
        assert info["pytorch_version"] is None

    def test_cuda_keys_present_when_cuda(self):
        """Chaves específicas de CUDA presentes quando device é CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA não disponível neste ambiente")
        info = dev.get_device_info()
        cuda_keys = {"cuda_version", "cudnn_version", "gpu_name",
                     "gpu_count", "total_memory_gb"}
        assert cuda_keys.issubset(info.keys()), (
            f"Chaves CUDA ausentes: {cuda_keys - info.keys()}"
        )

    def test_total_memory_gb_positive(self):
        """Memória total da GPU é um número positivo."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA não disponível neste ambiente")
        info = dev.get_device_info()
        assert info["total_memory_gb"] > 0

    def test_gpu_count_at_least_one(self):
        """gpu_count é pelo menos 1 quando CUDA disponível."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA não disponível neste ambiente")
        info = dev.get_device_info()
        assert info["gpu_count"] >= 1


# ---------------------------------------------------------------------------
# set_device_optimization()
# ---------------------------------------------------------------------------

class TestSetDeviceOptimization:
    """Testes para set_device_optimization()."""

    def test_does_not_raise_with_cpu(self):
        """Não lança exceção para device CPU."""
        dev.set_device_optimization(_cpu_device())

    def test_does_not_raise_with_none_device(self):
        """Não lança exceção quando device é None."""
        dev.set_device_optimization(None)

    def test_does_not_raise_when_torch_none(self):
        """Não lança exceção quando torch não está disponível."""
        with patch.object(dev, "torch", None):
            dev.set_device_optimization(_cpu_device())

    def test_cuda_enables_tf32(self):
        """Habilita TF32 e cuDNN benchmark para device CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA não disponível neste ambiente")
        dev.set_device_optimization(_cuda_device())
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cudnn.benchmark is True

    def test_mps_does_not_raise(self):
        """Não lança exceção para device MPS (experimental)."""
        dev.set_device_optimization(torch.device("mps"))


# ---------------------------------------------------------------------------
# get_torch_device()
# ---------------------------------------------------------------------------

class TestGetTorchDevice:
    """Testes para get_torch_device()."""

    _REQUIRED_KEYS = {"type", "name", "device"}
    _VALID_TYPES = {"GPU", "CPU", "TPU", "unavailable"}

    def test_returns_dict(self):
        """Retorno é um dicionário."""
        assert isinstance(dev.get_torch_device(), dict)

    def test_required_keys_present(self):
        """Dicionário sempre contém 'type', 'name' e 'device'."""
        result = dev.get_torch_device()
        assert self._REQUIRED_KEYS.issubset(result.keys()), (
            f"Chaves ausentes: {self._REQUIRED_KEYS - result.keys()}"
        )

    def test_type_is_valid(self):
        """Campo 'type' é um dos valores reconhecidos."""
        result = dev.get_torch_device()
        assert result["type"] in self._VALID_TYPES, (
            f"Tipo inesperado: {result['type']!r}"
        )

    def test_gpu_branch_when_cuda_available(self):
        """Retorna type='GPU' e device cuda quando CUDA disponível."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_name", return_value="Mock GPU"):
            result = dev.get_torch_device()
        assert result["type"] == "GPU"
        assert result["device"].type == "cuda"  # type: ignore[union-attr]
        assert result["name"] == "Mock GPU"

    def test_cpu_branch_when_no_cuda(self):
        """Retorna type='CPU' e device cpu quando CUDA indisponível."""
        with patch("torch.cuda.is_available", return_value=False), \
             patch.object(dev, "_XLA_AVAILABLE", False):
            result = dev.get_torch_device()
        assert result["type"] == "CPU"
        assert result["device"].type == "cpu"  # type: ignore[union-attr]

    def test_unavailable_when_torch_none(self):
        """Retorna type='unavailable' quando torch não está disponível."""
        with patch.object(dev, "torch", None), \
             patch.object(dev, "_XLA_AVAILABLE", False), \
             patch.object(dev, "xm", None):
            result = dev.get_torch_device()
        assert result["type"] == "unavailable"
        assert result["device"] is None

    def test_tpu_branch_when_xla_available(self):
        """Retorna type='TPU' quando XLA está disponível com dispositivos."""
        mock_xm = MagicMock()
        mock_xm.get_xla_supported_devices.return_value = ["tpu:0"]
        mock_xm.xla_device_kind.return_value = "TPU"
        mock_xm.xla_device.return_value = MagicMock()

        with patch.object(dev, "xm", mock_xm), \
             patch.object(dev, "_XLA_AVAILABLE", True):
            result = dev.get_torch_device()
        assert result["type"] == "TPU"
        assert result["name"] == "TPU"
