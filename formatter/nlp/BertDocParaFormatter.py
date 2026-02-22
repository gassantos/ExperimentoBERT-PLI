# -*- coding: utf-8 -*-
"""Formatador para a extração de embeddings com BertPoolOutMax (Etapa 3 do BERT-PLI).

Para cada par caso-decisão, cria uma matriz de tokenizações
``[M, N, L]`` com todos os pares de parágrafos (qi, ci), onde:

- *M* = número de parágrafos da query (``max_para_q``)
- *N* = número de parágrafos do documento candidato (``max_para_c``)
- *L* = comprimento máximo de sequência (``max_seq_length``)

Linhas/colunas faltantes são preenchidas com zeros.
"""
__author__ = 'yshao'

import torch

from transformers import AutoTokenizer

from formatter.Basic import BasicFormatter
from .bert_feature_tool import example_item_to_feature


class BertDocParaFormatter(BasicFormatter):
    """Formata batches para o modelo :class:`model.nlp.BertPoolOutMax.BertPoolOutMax`."""

    def __init__(self, config, mode, *args, **params):
        """Inicializa tokenizador e limites de parágrafos.

        Carrega ``AutoTokenizer`` a partir de ``bert_path`` e lê
        ``max_seq_length``, ``max_para_c`` e ``max_para_q`` da configuração.
        """
        super().__init__(config, mode, *args, **params)
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model", "bert_path"))
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode
        self.output_mode = config.get('model', 'output_mode')
        self.max_para_c = config.getint('model', 'max_para_c')
        self.max_para_q = config.getint('model', 'max_para_q')

    def process(self, data, config, mode, *args, **params):
        """Tokeniza todos os pares (qi, ci) e monta tensores 4-D.

        Para cada exemplo do batch gera uma grade de tokens de tamanho
        ``[M, N, L]``.  Linhas ou colunas ausentes (documentos com menos
        parágrafos que o máximo) são preenchidas com tensores zero.

        Args:
            data: lista de dicionários com ``guid``, ``label``, ``q_paras``
                e ``c_paras``.
            config: configuração do experimento.
            mode: ``'train'``, ``'valid'`` ou ``'test'``.

        Returns:
            Dicionário com tensores ``input_ids``, ``attention_mask`` e
            ``token_type_ids`` de forma ``[B, M, N, L]``, mais ``guid`` e
            ``label`` (exceto em ``test``).
        """
        # query_para_num=m, doc_para_num = n, matrix = m * n
        guids = []
        input_ids = []
        attention_mask = []
        token_type_ids = []
        if mode != 'test':
            labels = []

        for temp in data:
            guid = temp['guid']
            label = temp['label']
            q_paras = temp['q_paras']
            c_paras = temp['c_paras']
            input_ids_item = []
            attention_mask_item = []
            token_type_ids_item = []
            for m in range(min(self.max_para_q, len(q_paras))):
                q_p = q_paras[m]
                input_ids_row = []
                attention_mask_row = []
                token_type_ids_row = []
                for n in range(min(self.max_para_c, len(c_paras))):
                    c_p = c_paras[n]
                    example = {
                        'text_a': q_p,
                        'text_b': c_p,
                        'label': label
                    }
                    res_dict = example_item_to_feature(example, self.max_len, self.tokenizer, self.output_mode,
                                                       mode=mode, cls_token_at_end=False, pad_on_left=False,
                                                       cls_token_segment_id=0, pad_token_segment_id=0)
                    input_ids_row.append(res_dict['input_ids'])
                    attention_mask_row.append(res_dict['input_mask'])
                    token_type_ids_row.append(res_dict['segment_ids'])
                if len(c_paras) < self.max_para_c:
                    for j in range(len(c_paras), self.max_para_c):
                        input_ids_row.append([0] * self.max_len)
                        attention_mask_row.append([0] * self.max_len)
                        token_type_ids_row.append([0] * self.max_len)
                assert (len(input_ids_row) == self.max_para_c)
                assert (len(attention_mask_row) == self.max_para_c)
                assert (len(token_type_ids_row) == self.max_para_c)
                input_ids_item.append(input_ids_row)
                attention_mask_item.append(attention_mask_row)
                token_type_ids_item.append(token_type_ids_row)
            if len(q_paras) < self.max_para_q:
                for i in range(len(q_paras), self.max_para_q):
                    input_ids_row = []
                    attention_mask_row = []
                    token_type_ids_row = []
                    for j in range(self.max_para_c):
                        input_ids_row.append([0] * self.max_len)
                        attention_mask_row.append([0] * self.max_len)
                        token_type_ids_row.append([0] * self.max_len)
                    input_ids_item.append(input_ids_row)
                    attention_mask_item.append(attention_mask_row)
                    token_type_ids_item.append(token_type_ids_row)
            assert (len(input_ids_item) == self.max_para_q)
            assert (len(attention_mask_item) == self.max_para_q)
            assert (len(token_type_ids_item) == self.max_para_q)

            guids.append(guid)
            input_ids.append(input_ids_item)
            attention_mask.append(attention_mask_item)
            token_type_ids.append(token_type_ids_item)

            if mode != 'test':
                labels.append(label)

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        token_type_ids = torch.LongTensor(token_type_ids)

        # print('tensor size: ', input_ids.size(), attention_mask.size(), token_type_ids.size())
        # input('continue?')

        if mode != 'test':
            labels = torch.LongTensor(labels)

        if mode != 'test':
            return {'guid': guids, 'input_ids': input_ids, 'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'label': labels}
        else:
            return {'guid': guids, 'input_ids': input_ids, 'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids}

