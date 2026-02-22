"""
Módulo de utilitários para o projeto BERT-PLI.
Fornece funcionalidades multiplataforma para:
- Detecção e configuração de dispositivos (CPU/GPU)
- Reprodutibilidade de experimentos
- Gerenciamento de caminhos e diretórios

Os submódulos são carregados sob demanda (lazy import — PEP 562) para evitar
dupla inicialização quando um submódulo é executado diretamente via
``python -m utils.<submódulo>``.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Mapeamento símbolo → submódulo relativo (usado pelo __getattr__ lazy)
_LAZY: dict[str, str] = {
    # device
    "get_device":              ".device",
    "get_device_info":         ".device",
    "set_device_optimization": ".device",
    # seed
    "set_seed":                   ".seed",
    "ensure_reproducibility":     ".seed",
    "get_reproducibility_info":   ".seed",
    # paths
    "PathManager":             ".paths",
    # config
    "create_config":           ".config",
    "ConfigParser":            ".config",
    # reader
    "init_dataset":            ".reader",
    "init_test_dataset":       ".reader",
    "init_formatter":          ".reader",
}

# Importações estáticas apenas para type checkers (não executadas em runtime)
if TYPE_CHECKING:
    from .device import get_device, get_device_info, set_device_optimization
    from .seed import set_seed, ensure_reproducibility, get_reproducibility_info
    from .paths import PathManager
    from .config import create_config, ConfigParser
    from .reader import init_dataset, init_test_dataset, init_formatter


def __getattr__(name: str):
    """Carrega o símbolo *name* a partir do submódulo mapeado em ``_LAZY``.

    Implementa lazy imports conforme PEP 562 (Python ≥ 3.7): os submódulos
    só são importados quando um de seus símbolos é acessado pela primeira vez,
    evitando efeitos colaterais na inicialização do pacote.
    """
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name], __name__)
        obj = getattr(module, name)
        # Armazena no namespace do pacote para acesso O(1) nas próximas chamadas
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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