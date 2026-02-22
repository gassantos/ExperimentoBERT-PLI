"""
utils/reader.py — Inicialização de DataLoaders
================================================
Responsável por construir os ``DataLoader`` do PyTorch para os modos
``train``, ``valid`` e ``test``, combinando datasets e formatters
configurados via arquivo ``.config``.
"""

import logging
import platform

from torch.utils.data import DataLoader

import formatter as form
from dataset import dataset_list

logger = logging.getLogger(__name__)

collate_fn = {}
formatter = {}


class CollateFn:
    """Wrapper picklable para ``collate_fn``, necessário no multiprocessing do Windows."""

    def __init__(self, formatter_obj, config, mode):
        """Args:
            formatter_obj: Instância do formatter responsável por processar os dados.
            config: ConfigParser com os parâmetros do experimento.
            mode: Modo de operação (``'train'``, ``'valid'`` ou ``'test'``).
        """
        self.formatter_obj = formatter_obj
        self.config = config
        self.mode = mode

    def __call__(self, data):
        """Processa um batch de dados aplicando o formatter configurado.

        Args:
            data: Batch bruto retornado pelo ``Dataset``.

        Returns:
            Batch formatado pronto para o modelo.
        """
        return self.formatter_obj.process(data, self.config, self.mode)


def init_formatter(config, task_list, *args, **params):
    """Inicializa formatters para as tarefas especificadas e constrói funções de collate."""
    for task in task_list:
        formatter[task] = form.init_formatter(config, task, *args, **params)
        collate_fn[task] = CollateFn(formatter[task], config, task)


def init_one_dataset(config, mode, *args, **params):
    """ Inicializa e retorna um DataLoader para um dataset específico baseado no modo fornecido.
        config (ConfigParser): Objeto de configuração contendo parâmetros do dataset e treinamento.
        mode (str): Modo de operação. Valores possíveis: 'train', 'valid' ou 'test'.
        *args: Argumentos posicionais adicionais repassados ao dataset.
        **params: Argumentos nomeados adicionais repassados ao dataset.
        NotImplementedError: Se o tipo de dataset especificado na configuração não estiver registrado.
        DataLoader: Instância do DataLoader configurada com os parâmetros do modo especificado.
    Notes:
        - Para os modos 'valid' e 'test', tenta carregar configurações específicas da seção [eval].
          Caso não encontradas, utiliza os valores padrão da seção [train].
        - Em sistemas Windows, o uso de múltiplos workers (reader_num > 0) é desabilitado
          automaticamente para evitar erros de serialização no multiprocessing.
    """
    temp_mode = mode
    if mode != "train":
        try:
            config.get("data", "%s_dataset_type" % temp_mode)
        except Exception:
            logger.warning(
                "[reader] %s_dataset_type has not been defined in config file, "
                "use [dataset] train_dataset_type instead." % temp_mode
            )
            temp_mode = "train"

    which = config.get("data", "%s_dataset_type" % temp_mode)

    if which not in dataset_list:
        logger.error("There is no dataset called %s, check your config." % which)
        raise NotImplementedError

    dataset = dataset_list[which](config, mode, *args, **params)
    batch_size = config.getint("train", "batch_size")
    shuffle = config.getboolean("train", "shuffle")
    reader_num = config.getint("train", "reader_num")
    drop_last = True

    if mode in ["valid", "test"]:
        drop_last = False
        try:
            batch_size = config.getint("eval", "batch_size")
        except Exception:
            logger.warning("[eval] batch size has not been defined in config file, use [train] batch_size instead.")
        try:
            shuffle = config.getboolean("eval", "shuffle")
        except Exception:
            shuffle = False
            logger.warning("[eval] shuffle has not been defined in config file, use false as default.")
        try:
            reader_num = config.getint("eval", "reader_num")
        except Exception:
            logger.warning("[eval] reader num has not been defined in config file, use [train] reader num instead.")

    # Windows: multiprocessing DataLoader causa erros de serialização
    if platform.system() == "Windows" and reader_num > 0:
        logger.warning(
            "[reader] Multiprocessing DataLoader (reader_num > 0) can cause issues on Windows. "
            "Setting num_workers=0. Use Linux/macOS to enable multiprocessing."
        )
        reader_num = 0

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=reader_num,
        collate_fn=collate_fn[mode],
        drop_last=drop_last,
    )


def init_test_dataset(config, *args, **params):
    """
    Inicializa e retorna um dataset de teste específico baseado na configuração fornecida.

    Args:
        config (_type_): Objeto de configuração contendo parâmetros do dataset de teste.
        *args: Argumentos posicionais adicionais repassados ao dataset de teste.
        **params: Argumentos nomeados adicionais repassados ao dataset de teste.

    Returns:
        _type_: Instância do dataset de teste configurado com os parâmetros fornecidos.
    """
    init_formatter(config, ["test"], *args, **params)
    return init_one_dataset(config, "test", *args, **params)


def init_dataset(config, *args, **params):
    """Inicializa e retorna os datasets de treinamento e validação com base na configuração fornecida.

    Args:
        config (_type_): Objeto de configuração contendo parâmetros dos datasets de treinamento e validação.
        *args: Argumentos posicionais adicionais repassados aos datasets de treinamento e validação.
        **params: Argumentos nomeados adicionais repassados aos datasets de treinamento e validação.

    Returns:
        _type_: Tupla contendo as instâncias dos datasets de treinamento e validação configurados com os parâmetros fornecidos.
    """
    init_formatter(config, ["train", "valid"], *args, **params)
    train_dataset = init_one_dataset(config, "train", *args, **params)
    valid_dataset = init_one_dataset(config, "valid", *args, **params)
    return train_dataset, valid_dataset
