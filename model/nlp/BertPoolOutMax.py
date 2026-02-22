# -*- coding: utf-8 -*-
"""model/nlp/BertPoolOutMax.py — Extrator de embeddings BERT com max-pooling
==========================================================================
Estágio 3 do pipeline BERT-PLI: percorre todos os pares padrão-parágrafo
(query × candidato) sem gradiente, extraí o ``pooler_output`` do BERT
para cada par e aplica ``MaxPool2d`` ao longo da dimensão de parágrafos
candidatos, produzindo um vetor de interação por parágrafo da query.

O resultado é uma lista de interações que será consumida pelo
:class:`AttentionRNN` no estágio seguinte.
"""
__author__ = 'yshao'


import torch
import torch.nn as nn
from transformers import BertModel
import logging
logger = logging.getLogger(__name__)

class BertPoolOutMax(nn.Module):
    """Módulo de pooling de interações BERT em nível de parágrafo.

    Para cada amostra do batch, itera sobre os parágrafos da query em
    janelas de tamanho ``step``, processa cada janela com BERT sem gradiente
    (``torch.no_grad()``) e reduz a dimensão de parágrafos candidatos
    via ``MaxPool2d(kernel=(1, max_para_c))``.

    Parâmetros lidos do ``.config``:
        ``model.max_para_c``, ``model.max_para_q``, ``model.step``,
        ``data.max_seq_length``, ``model.bert_path``.
    """

    def __init__(self, config, gpu_list, *args, **params):
        """Inicializa BERT e a camada MaxPool2d.

        Args:
            config:   ConfigParser com parâmetros do modelo e dos dados.
            gpu_list: Lista de IDs de GPU (reservado para ``init_multi_gpu``).
        """
        super(BertPoolOutMax, self).__init__()
        self.max_para_c = config.getint('model', 'max_para_c')
        self.max_para_q = config.getint('model', 'max_para_q')
        self.step = config.getint('model', 'step')
        self.max_len = config.getint("data", "max_seq_length")
        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        # self.maxpool = nn.MaxPool1d(kernel_size=self.max_para_c)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, self.max_para_c))

    def init_multi_gpu(self, device, config, *args, **params) -> None:
        """Distribui o encoder BERT em múltiplas GPUs via ``nn.DataParallel``.

        Args:
            device: Lista de IDs de GPU passada ao ``DataParallel``.
            config: ConfigParser (não utilizado; mantido para interface uniforme).
        """
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data: dict, config, gpu_list, acc_result, mode: str) -> dict:
        """Extrai vetores de interação para cada par (query, documento).

        Args:
            data:       Dict com ``'input_ids'`` ``[B, max_para_q, max_para_c, L]``,
                        ``'attention_mask'``, ``'token_type_ids'`` e ``'guid'``.
            config:     ConfigParser (não utilizado no forward; mantido para interface uniforme).
            gpu_list:   Lista de IDs de GPU (não utilizado diretamente).
            acc_result: Acumulador de acurácia (não utilizado neste estágio).
            mode:       Modo de execução (``'train'``, ``'valid'`` ou ``'test'``).

        Returns:
            ``{'output': [(guid, q_lst), ...]}`` onde ``q_lst`` é uma lista de
            ``max_para_q`` vetores de interação (um por parágrafo da query).
        """
        input_ids, attention_mask, token_type_ids = data['input_ids'], data['attention_mask'], data['token_type_ids']
        with torch.no_grad():
            output = []
            for k in range(input_ids.size()[0]):
                q_lst = []
                for i in range(0, self.max_para_q, self.step):
                    # print(input_ids[k, i:i+self.step].view(-1, self.max_len).size())
                    _bert_out = self.bert(input_ids[k, i:i+self.step].view(-1, self.max_len),
                                           token_type_ids=token_type_ids[k, i:i+self.step].view(-1, self.max_len),
                                           attention_mask=attention_mask[k, i:i+self.step].view(-1, self.max_len))
                    lst = _bert_out.pooler_output
                    # print('before view', lst.size())
                    lst = lst.view(self.step, self.max_para_c, -1)
                    # print('after view', lst.size())
                    lst = lst.permute(2, 0, 1)
                    lst = lst.unsqueeze(0)
                    max_out = self.maxpool(lst)
                    max_out = max_out.squeeze()
                    if max_out.dim() > 1:
                        max_out = max_out.transpose(0, 1)
                    else:
                        # If max_out is 1D, reshape it
                        max_out = max_out.unsqueeze(0)
                    q_lst.extend(max_out.cpu().tolist())
                    #input('continue?')
                # print(len(q_lst))
                #exit()
                assert (len(q_lst) == self.max_para_q)
                output.append([data['guid'][k], q_lst])
            return {"output": output}
        
