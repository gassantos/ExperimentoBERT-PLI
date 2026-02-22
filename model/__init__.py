"""model — Registro central de modelos PyTorch do projeto BERT-PLI
==============================================================
Exporta os três modelos do pipeline e a função de fábrica :func:`get_model`.

Pipeline de dois estágios:

- **Estágio 2** — :class:`BertPoint`: fine-tuning supervisionado par a par de parágrafos.
- **Estágio 3** — :class:`BertPoolOutMax`: extrator de embeddings (poolout) sem gradiente.
- **Estágio 4** — :class:`AttentionRNN`: agregação temporal com GRU/LSTM + Atenção.
"""
from .nlp.BertPoint import BertPoint
from .nlp.BertPoolOutMax import BertPoolOutMax
from .nlp.AttenRNN import AttentionRNN

model_list = {
    "BertPoint": BertPoint,
    "BertPoolOutMax": BertPoolOutMax,
    "AttenRNN": AttentionRNN
}


def get_model(model_name: str) -> type:
    """Retorna a classe do modelo correspondente ao nome fornecido.

    Args:
        model_name: Um dos valores registrados em ``model_list``
            (``'BertPoint'``, ``'BertPoolOutMax'``, ``'AttenRNN'``).

    Returns:
        Classe do modelo (não instanciada).

    Raises:
        NotImplementedError: Se ``model_name`` não estiver em ``model_list``.
    """
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
