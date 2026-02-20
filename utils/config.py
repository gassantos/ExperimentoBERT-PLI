"""
utils/config.py — Parser de configuração hierárquico
======================================================
Envolve o ``configparser.RawConfigParser`` da stdlib com suporte a três
camadas de configuração (fallback em cascata):

1. ``config/default.config`` — valores base do projeto
2. ``config/default_local.config`` — overrides locais (se existir)
3. Arquivo de experimento passado pelo usuário — overrides finais
"""

import configparser
import functools
import os


class ConfigParser:
    """
    Parser de configuração com fallback em cascata.

    Resolve cada chave na ordem: config do experimento → local → default.
    """

    def __init__(self, *args, **params):
        self.default_config = configparser.RawConfigParser(*args, **params)
        self.local_config = configparser.RawConfigParser(*args, **params)
        self.config = configparser.RawConfigParser(*args, **params)

    def read(self, filenames, encoding=None):
        if os.path.exists("config/default_local.config"):
            self.local_config.read("config/default_local.config", encoding=encoding)
        else:
            self.local_config.read("config/default.config", encoding=encoding)

        self.default_config.read("config/default.config", encoding=encoding)
        self.config.read(filenames, encoding=encoding)


def _build_func(func_name):
    @functools.wraps(getattr(configparser.RawConfigParser, func_name))
    def func(self, *args, **kwargs):
        try:
            return getattr(self.config, func_name)(*args, **kwargs)
        except Exception:
            try:
                return getattr(self.local_config, func_name)(*args, **kwargs)
            except Exception:
                return getattr(self.default_config, func_name)(*args, **kwargs)

    return func


def create_config(path: str) -> ConfigParser:
    """
    Instancia e retorna um :class:`ConfigParser` carregado com o arquivo
    de configuração especificado.

    Args:
        path: Caminho para o arquivo ``.config`` do experimento.

    Returns:
        Instância de :class:`ConfigParser` pronta para uso.
    """
    for func_name in dir(configparser.RawConfigParser):
        if not func_name.startswith("_") and func_name != "read":
            setattr(ConfigParser, func_name, _build_func(func_name))

    config = ConfigParser()
    config.read(path)
    return config
