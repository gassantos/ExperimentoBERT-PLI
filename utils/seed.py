"""
Módulo para garantir reprodutibilidade em experimentos.
Configura seeds para todas as bibliotecas e frameworks usados.
"""
import random
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    torch = None
    _TORCH_AVAILABLE = False
    logger.warning(f"PyTorch não disponível: {e}")


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Define seeds para reprodutibilidade em todas as plataformas.
    
    Args:
        seed: Seed para geração de números aleatórios
        deterministic: Se True, força operações determinísticas (pode reduzir performance)
    """
    logger.info(f"Setting random seed to {seed} (deterministic={deterministic})")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Hash seed para Python (importante para dicionários)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch (se disponível)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        
        # CUDA (se disponível)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # Para multi-GPU
            
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                logger.info("CUDA deterministic mode enabled (may reduce performance)")
            else:
                torch.backends.cudnn.benchmark = True
                logger.info("CUDA benchmark mode enabled (better performance, not fully deterministic)")
        
        # Apple Silicon MPS (se disponível)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
            logger.info("MPS seed set")
    
    # Transformers (se disponível)
    try:
        from transformers import set_seed as transformers_set_seed
        transformers_set_seed(seed)
        logger.info("Transformers seed set")
    except ImportError:
        pass


def get_reproducibility_info():
    """
    Retorna informações sobre configurações de reprodutibilidade.
    
    Returns:
        dict: Estado atual das configurações de reprodutibilidade
    """
    info = {
        "python_hash_seed": os.environ.get('PYTHONHASHSEED', 'not set'),
        "numpy_seed_available": True,
        "torch_seed_available": _TORCH_AVAILABLE,
    }
    
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        info.update({
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        })
    
    if _TORCH_AVAILABLE and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps_available"] = True
    
    return info


def ensure_reproducibility(seed: int = 42):
    """
    Wrapper conveniente para garantir reprodutibilidade completa.
    Usa configurações mais restritivas.
    
    Args:
        seed: Seed para geração de números aleatórios
    """
    set_seed(seed, deterministic=True)
    
    # Configurações adicionais para frameworks específicos
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Para operações CUDA determinísticas
    
    # Aviso sobre performance
    logger.warning(
        "Full reproducibility mode enabled. This may significantly reduce performance. "
        "For production, consider using deterministic=False in set_seed()."
    )