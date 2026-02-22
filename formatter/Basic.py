"""Formatador base sem transformações, utilizado como fallback e para testes."""


class BasicFormatter:
    """Formatador identidade: devolve os dados de entrada sem modificações.

    Serve como classe base para os demais formatadores e como formatador
    padrão em pipelines sem pré-processamento específico.
    """

    def __init__(self, config, mode, *args, **params):
        """Armazena config e mode para acesso pelos subformatadores."""
        self.config = config
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
        """Retorna *data* sem transformações."""
        return data

