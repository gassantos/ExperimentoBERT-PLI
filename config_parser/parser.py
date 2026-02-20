import configparser
import os
import functools


class ConfigParser:
    """
    A configuration parser that manages multiple configuration files with support for default and local configurations.
    This class wraps Python's `configparser.RawConfigParser` to handle three levels of configuration:
    - **default**: Base configuration loaded from `config/default.config`.
    - **local**: Overrides loaded from `config/default_local.config` if it exists, otherwise falls back to `config/default.config`.
    - **custom**: Additional configuration loaded from user-specified files.
    Args:
        *args: Positional arguments passed to `configparser.RawConfigParser`.
        **params: Keyword arguments passed to `configparser.RawConfigParser`.
    Methods:
        read(filenames, encoding=None):
            Reads configuration files into their respective parsers.
            Args:
                filenames (str | list): Path(s) to the custom configuration file(s).
                encoding (str, optional): Encoding used to read the files. Defaults to None.
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
    for func_name in dir(configparser.RawConfigParser):
        if not func_name.startswith('_') and func_name != "read":
            setattr(ConfigParser, func_name, _build_func(func_name))

    config = ConfigParser()
    config.read(path)

    return config
