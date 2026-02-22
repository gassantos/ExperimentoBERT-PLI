"""model/optimizer.py — Fábrica de otimizadores PyTorch
=======================================================
Instancia o otimizador configurado no arquivo ``.config`` do experimento.

Otimizadores suportados (chave ``[train] optimizer``):

- ``adam``     — :class:`torch.optim.Adam`
- ``adamw``    — :class:`torch.optim.AdamW`
- ``sgd``      — :class:`torch.optim.SGD`
- ``bert_adam`` — alias para :class:`torch.optim.AdamW` (sucessor do BertAdam removido na
  ``transformers ≥4.46``); o warmup linear é gerenciado por
  ``get_linear_schedule_with_warmup`` em ``tools/train_tool.py``.
"""
import torch.optim as optim


def init_optimizer(model: "nn.Module", config, *args, **params) -> "torch.optim.Optimizer":
    """Instancia e retorna o otimizador especificado na configuração.

    Args:
        model:  Módulo PyTorch cujos parâmetros serão otimizados.
        config: ConfigParser com as chaves ``train.optimizer``,
                ``train.learning_rate`` e ``train.weight_decay``.

    Returns:
        Instância do otimizador pronta para uso no loop de treino.

    Raises:
        NotImplementedError: Se ``optimizer`` não for um dos valores suportados.
    """
    optimizer_type = config.get("train", "optimizer")
    learning_rate = config.getfloat("train", "learning_rate")
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "bert_adam":
        # torch.optim.AdamW é o sucessor moderno do BertAdam (transformers.AdamW removido na v4.46+).
        # O warmup linear é tratado por get_linear_schedule_with_warmup em train_tool.py.
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                weight_decay=config.getfloat("train", "weight_decay"))
    else:
        raise NotImplementedError

    return optimizer
