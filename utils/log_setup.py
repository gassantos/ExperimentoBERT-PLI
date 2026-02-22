"""utils/log_setup.py — Configuração centralizada de logging multiprocessing-safe.

Implementa o padrão ``QueueHandler + QueueListener`` (recomendado pela
documentação oficial do Python para ambientes multiprocessing) de forma que
**apenas o processo principal** escreve nos handlers reais (stdout e arquivo),
enquanto cada worker envia suas mensagens para uma fila compartilhada.

Arquitetura::

    Worker 0 ─┐
    Worker 1 ─┼──► QueueHandler ──► _LOG_QUEUE ──► QueueListener ──► StreamHandler (stdout)
    Worker N ─┘                                                   └──► FileHandler  (arquivo)

Loggers ruidosos de bibliotecas de terceiros (httpx, huggingface_hub, etc.)
são silenciados em todos os processos, deixando o log focado no experimento.

Uso típico::

    # Processo principal — uma única chamada
    from utils.log_setup import setup_main_logging, setup_worker_logging, _LOG_QUEUE

    listener = setup_main_logging(logfile="logs/experimento.log")
    try:
        with ProcessPoolExecutor(
            initializer=setup_worker_logging,
            initargs=(_LOG_QUEUE,),
        ) as executor:
            ...
    finally:
        listener.stop()
"""
from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
import sys
from pathlib import Path
from typing import Union

try:
    import transformers as _transformers
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

# ---------------------------------------------------------------------------
# Loggers de terceiros silenciados em todos os processos
# ---------------------------------------------------------------------------

# Reduzidos a WARNING: exibem erros reais mas suprimem ruído INFO/DEBUG
_NOISY_LOGGERS: tuple[str, ...] = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "huggingface_hub.file_download",
    "filelock",
    "urllib3",
    "urllib3.connectionpool",
    "transformers.tokenization_utils_base",
    "transformers.configuration_utils",
    "transformers.modeling_utils",
)

# Reduzidos a ERROR: emitem avisos operacionais irrelevantes em WARNING
# (ex.: "unauthenticated requests" do HF Hub sem HF_TOKEN definido)
_ERROR_ONLY_LOGGERS: tuple[str, ...] = (
    "huggingface_hub.utils._http",
    "huggingface_hub.utils._headers",
)

_LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Fila compartilhada: criada uma única vez no módulo e passada via initargs
# para cada worker — spawn-safe porque é serializada por valor.
_LOG_QUEUE: multiprocessing.Queue = multiprocessing.Queue(-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silence_noisy_loggers() -> None:
    """Eleva o nível dos loggers de terceiros em qualquer processo.

    * ``_NOISY_LOGGERS``      → WARNING  (suprime DEBUG/INFO ruidosos)
    * ``_ERROR_ONLY_LOGGERS`` → ERROR    (suprime avisos operacionais irrelevantes)
    * ``transformers``        → ERROR via API nativa, desabilita progress bar
      e remove o StreamHandler próprio (stderr) para evitar saída duplicada.
    """
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in _ERROR_ONLY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    if _HAS_TRANSFORMERS:
        # set_verbosity_error() silencia:
        #   - "BertModel LOAD REPORT" (WARNING do modeling_utils)
        #   - "The following layers were not sharded" (WARNING do tensor_parallel)
        # disable_progress_bar() para o tqdm "Loading weights: XX%"
        # disable_default_handler() remove o StreamHandler→stderr próprio do
        #   transformers (propagate=False por padrão).
        # enable_propagation() redireciona os ERRORs reais para o root logger
        #   (e portanto para o QueueHandler), evitando perda silenciosa.
        _transformers.logging.set_verbosity_error()
        _transformers.logging.disable_progress_bar()
        _transformers.logging.disable_default_handler()
        _transformers.logging.enable_propagation()


def _make_formatter() -> logging.Formatter:
    return logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def setup_main_logging(
    logfile: Union[Path, str],
    level: int = logging.INFO,
) -> logging.handlers.QueueListener:
    """Configura o logging do processo principal com QueueHandler + QueueListener.

    Deve ser chamada **uma única vez**, antes de iniciar os workers.  O root
    logger passa a enviar para ``_LOG_QUEUE``; o ``QueueListener`` consome a
    fila e despacha para os handlers reais (stdout + arquivo).

    Args:
        logfile: Caminho do arquivo de log (modo append).
        level: Nível mínimo de log para o root logger (padrão: INFO).

    Returns:
        Instância de ``QueueListener`` já iniciada.  O chamador deve invocar
        ``listener.stop()`` ao encerrar o programa para garantir que todas as
        mensagens pendentes na fila sejam escritas.
    """
    formatter = _make_formatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    file_handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    listener = logging.handlers.QueueListener(
        _LOG_QUEUE,
        stream_handler,
        file_handler,
        respect_handler_level=True,
    )
    listener.start()

    # Root logger apenas encaminha para a fila; o listener escreve nos handlers
    root = logging.getLogger()
    # Remove handlers anteriores (evita que basicConfig de bibliotecas externes duplique)
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.setLevel(level)
    root.addHandler(logging.handlers.QueueHandler(_LOG_QUEUE))

    _silence_noisy_loggers()
    return listener


def setup_worker_logging(queue: multiprocessing.Queue) -> None:
    """Configura o logging de um processo worker (spawn).

    Deve ser usada como ``initializer`` do ``ProcessPoolExecutor``::

        ProcessPoolExecutor(
            initializer=setup_worker_logging,
            initargs=(_LOG_QUEUE,),
        )

    O worker **não** abre handlers próprios — apenas envia mensagens para a
    fila do processo principal, que é o único responsável por escrever em
    stdout e no arquivo de log.

    Args:
        queue: Fila compartilhada criada pelo processo principal
            (normalmente ``_LOG_QUEUE`` de :mod:`utils.log_setup`).
    """
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    root.addHandler(logging.handlers.QueueHandler(queue))
    _silence_noisy_loggers()
