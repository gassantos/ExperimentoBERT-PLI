"""Pacote de formatadores (collate) do projeto BERT-PLI.

Cada formatador recebe um batch de exemplos brutos e devolve tensores
prontos para o modelo.  O registro ``formatter_list`` mapeia o nome
configurado em ``[data] train_formatter_type`` para a classe concreta.

Formatadores disponíveis:

- **Basic**: passa os dados sem transformação (baseline).
- **BertPairText**: tokeniza pares de texto para BertPoint (Etapa 2).
- **BertDocPara**: tokeniza pares parágrafo × parágrafo para BertPoolOutMax
  (Etapa 3), gerando tensores 4-D ``[B, M, N, L]``.
- **AttenRNN**: converte embeddings pré-computados em tensores para AttentionRNN
  (Etapa 4).
"""
import logging

from .Basic import BasicFormatter
from .nlp.BertPairTextFormatter import BertPairTextFormatter
from .nlp.BertDocParaFormatter import BertDocParaFormatter
from .nlp.AttenRNNFormatter import AttenRNNFormatter

logger = logging.getLogger(__name__)

formatter_list = {
    "Basic": BasicFormatter,
    "BertPairText": BertPairTextFormatter,
    "BertDocPara": BertDocParaFormatter,
    'AttenRNN': AttenRNNFormatter
}


def init_formatter(config, mode, *args, **params):
    """Instancia o formatador adequado para a partição *mode*.

    Lê a chave ``{mode}_formatter_type`` da secção ``[data]``; se ausente,
    usa ``train_formatter_type`` como fallback (comportamento de ``valid``/
    ``test`` que não definem seu próprio tipo).

    Args:
        config: instância de :class:`utils.config.ConfigParser`.
        mode: partição — ``'train'``, ``'valid'`` ou ``'test'``.
        *args: argumentos adicionais repassados ao construtor do formatador.
        **params: parâmetros adicionais repassados ao construtor.

    Returns:
        Instância do formatador selecionado.

    Raises:
        Exception: se o tipo de formatador não estiver registrado em
            ``formatter_list``.
    """
    temp_mode = mode
    if mode != "train":
        try:
            config.get("data", "%s_formatter_type" % temp_mode)
        except Exception as e:
            logger.warning(
                "[reader] %s_formatter_type has not been defined in config file, use [dataset] train_formatter_type instead." % temp_mode)
            temp_mode = "train"
    which = config.get("data", "%s_formatter_type" % temp_mode)

    if which in formatter_list:
        formatter = formatter_list[which](config, mode, *args, **params)

        return formatter
    else:
        logger.error("There is no formatter called %s, check your config." % which)
        raise NotImplementedError
