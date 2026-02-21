"""
Módulo de utilitários para o projeto BERT-PLI.
Fornece funcionalidades multiplataforma para:
- Detecção e configuração de dispositivos (CPU/GPU)
- Reprodutibilidade de experimentos
- Gerenciamento de caminhos e diretórios
"""

from .device import get_device, get_device_info, set_device_optimization
from .seed import set_seed, ensure_reproducibility, get_reproducibility_info
from .paths import PathManager
from .config import create_config, ConfigParser
from .reader import init_dataset, init_test_dataset, init_formatter

__all__ = [
    # Device utilities
    'get_device',
    'get_device_info',
    'set_device_optimization',
    
    # Reproducibility utilities
    'set_seed',
    'ensure_reproducibility',
    'get_reproducibility_info',
    
    # Path utilities
    'PathManager',

    # Config utilities
    'create_config',
    'ConfigParser',

    # Reader utilities
    'init_dataset',
    'init_test_dataset',
    'init_formatter',
]

__version__ = '0.1.0'