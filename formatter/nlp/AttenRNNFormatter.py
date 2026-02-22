# -*- coding: utf-8 -*-
"""Formatador para a etapa de agregação com AttentionRNN (Etapa 4 do BERT-PLI).

Converte embeddings pré-computados (saída do pool-out) em tensores
``[B, M, H]`` onde *B* é o tamanho do batch, *M* é o número máximo de
parágrafos da query (``max_para_q``) e *H* é a dimensão do embedding.

Espera exemplos com as chaves:
  - ``guid``: identificador do par caso-decisão.
  - ``res``: lista de ``max_para_q`` embeddings de tamanho fixo.
  - ``label``: rótulo inteiro (ausente no modo ``test``).
"""
__author__ = 'yshao'

import json
import torch
import os

from formatter.Basic import BasicFormatter


class AttenRNNFormatter(BasicFormatter):
    """Formata um batch de embeddings para o modelo :class:`model.nlp.AttenRNN.AttentionRNN`."""

    def __init__(self, config, mode, *agrs, **params):
        """Lê ``max_para_q`` da secção ``[model]`` para validar o tamanho do embedding."""
        super().__init__(config, mode, *agrs, **params)
        self.max_para_q = config.getint('model', 'max_para_q')
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
        """Converte um batch de exemplos em tensores para AttentionRNN.

        Args:
            data: lista de dicionários com chaves ``guid``, ``res`` e,
                fora do modo ``test``, ``label``.
            config: configuração do experimento (não utilizado diretamente).
            mode: ``'train'``, ``'valid'`` ou ``'test'``.

        Returns:
            Dicionário com:
              - ``guid``: lista de identificadores.
              - ``input``: :class:`torch.Tensor` ``[B, M, H]``.
              - ``label``: :class:`torch.LongTensor` ``[B]`` (ausente em ``test``).
        """
        inputs = []
        guids = []
        if mode != 'test':
            labels = []

        for temp in data:
            # guid = temp['id_']
            guid = temp['guid']
            emb_mtx = temp['res']
            assert (len(emb_mtx) == self.max_para_q)
            inputs.append(emb_mtx)
            guids.append(guid)

            if mode != 'test':
                labels.append(temp['label'])

        inputs = torch.tensor(inputs)

        if mode != 'test':
            labels = torch.LongTensor(labels)
            return {'guid': guids, 'input': inputs, 'label': labels}
        else:
            return {'guid': guids, 'input': inputs}

