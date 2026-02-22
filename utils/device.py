"""
Módulo para detecção automática de dispositivo de computação.
Suporta CUDA (NVIDIA), TPU (TorchXLA), MPS (Apple Silicon) e CPU.
"""
from __future__ import annotations

import platform
import logging
from typing import TYPE_CHECKING

# Importações apenas para o analisador de tipos (Pylance/mypy).
# Em runtime TYPE_CHECKING é sempre False, portanto estes imports
# não geram custo de inicialização nem dependência obrigatória.
if TYPE_CHECKING:
    import torch
    import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

try:
    import torch  # type: ignore[no-redef]

    try:
        import torch_xla.core.xla_model as xm  # type: ignore[no-redef]
        _XLA_AVAILABLE = True
    except (ImportError, OSError) as e:
        xm = None  # type: ignore[assignment]
        _XLA_AVAILABLE = False
        logger.warning(f"Torch_XLA não disponível: {e}")

except (ImportError, OSError):
    torch = None  # type: ignore[assignment]
    logger.warning("PyTorch não disponível neste ambiente")


def get_device(prefer_cpu: bool = False):
    """
    Detecta o melhor dispositivo disponível de forma multiplataforma.
    
    Args:
        prefer_cpu: Se True, força uso de CPU mesmo com GPU disponível
    
    Returns:
        torch.device: Device otimizado para a plataforma atual, ou None se torch indisponível
    """
    if torch is None:
        logger.warning("PyTorch não disponível. Retornando device=None.")
        return None

    if prefer_cpu:
        logger.info("CPU mode forced by user")
        return torch.device("cpu")
    
    system = platform.system()
    
    # Windows/Linux com CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using CUDA GPU: {gpu_name} ({gpu_memory:.2f} GB)")
        return device
    
    # macOS com Apple Silicon (MPS)
    elif system == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")
        return device
    
    # CPU fallback
    else:
        device = torch.device("cpu")
        logger.info(f"Using CPU on {system} system")
        if system == "Darwin":
            logger.warning("MPS not available. Ensure PyTorch version supports Apple Silicon.")
        elif system in ["Windows", "Linux"]:
            logger.warning("CUDA not available. Install CUDA toolkit for GPU acceleration.")
        return device


def get_device_info():
    """
    Retorna informações detalhadas sobre o dispositivo.
    
    Returns:
        dict: Informações sobre dispositivo, memória e capacidade
    """
    if torch is None:
        return {
            "device_type": "unavailable",
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "pytorch_version": None,
            "error": "PyTorch não disponível neste ambiente"
        }
    device = get_device()
    assert device is not None  # torch is not None, portanto get_device() sempre retorna um device
    info = {
        "device_type": device.type,
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }
    
    if device.type == "cuda":
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        })
    elif device.type == "mps":
        info.update({
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
        })
    
    return info


def set_device_optimization(device):
    """
    Configura otimizações específicas do dispositivo.
    
    Args:
        device: torch.device para otimizar
    """
    if torch is None or device is None:
        logger.warning("PyTorch não disponível. Otimizações de device ignoradas.")
        return
    if device.type == "cuda":
        # Habilita TF32 para Ampere GPUs (melhor performance)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Habilita benchmark para encontrar melhor algoritmo
        torch.backends.cudnn.benchmark = True
        logger.info("CUDA optimizations enabled (TF32, cuDNN benchmark)")
    elif device.type == "mps":
        # MPS ainda é experimental, sem otimizações específicas por enquanto
        logger.info("MPS device set (experimental support)")
    else:
        logger.info("CPU mode - no specific optimizations applied")


def get_torch_device() -> dict:
    """Retorna o dispositivo PyTorch disponível (CPU, GPU ou TPU)."""
    if xm is not None and _XLA_AVAILABLE and len(xm.get_xla_supported_devices()) > 0:
        return {
            'type': 'TPU',
            'name': xm.xla_device_kind(),
            'device': xm.xla_device()
        }
    if torch is not None and torch.cuda.is_available():
        return {
            'type': 'GPU',
            'name': torch.cuda.get_device_name(0),
            'device': torch.device('cuda')
        }
    if torch is not None:
        return {
            'type': 'CPU',
            'name': platform.processor(),
            'device': torch.device('cpu')
        }
    return {'type': 'unavailable', 'name': None, 'device': None}