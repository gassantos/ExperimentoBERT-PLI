# -*- coding: utf-8 -*-
"""model/nlp/BertPoint.py — Fine-tuning BERT par a par de parágrafos
====================================================================
Estágio 2 do pipeline BERT-PLI: recebe um par de parágrafos (query, candidato)
como sequência única e fine-tunes um BERT pré-treinado para classificação
binaria de entailment.

Também atua como extrator de embeddings quando ``output.pool_out = true``:
neste caso, o ``forward`` retorna o mean-pool de ``last_hidden_state``
sem passar pela camada de classificação.
"""
__author__ = 'yshao'

import torch
import torch.nn as nn
from transformers import BertModel

from tools.accuracy_init import init_accuracy_function


class BertPoint(nn.Module):
    """Classificador de entailment baseado em BERT com fine-tuning.

    Codifica o par de parágrafos (``[CLS] q [SEP] c [SEP]``) com BERT e passa
    o mean-pool da última camada por uma camada linear de classificação.

    Quando ``output.pool_out = true``, o ``forward`` retorna apenas o vetor
    de embedding antes da camada FC para ser consumido pelo estágio seguinte.

    Parâmetros lidos do ``.config``:
        ``model.output_dim``, ``model.output_mode`` (``'classification'`` ou ``'regression'``),
        ``model.bert_path``.
    """

    def __init__(self, config, gpu_list, *args, **params):
        """Inicializa BERT, camada FC e função de perda/acurácia.

        Args:
            config:   ConfigParser com parâmetros do modelo.
            gpu_list: Lista de IDs de GPU (reservado para ``init_multi_gpu``).
        """
        super(BertPoint, self).__init__()

        self.output_dim = config.getint("model", "output_dim")
        self.output_mode = config.get('model', 'output_mode')

        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        # Lê hidden_size diretamente do backbone carregado — compatível com
        # bert-base (768), bert-large (1024), DeBERTa, RoBERTa, LegalBERT, etc.
        self.hidden_size = self.bert.config.hidden_size
        self.fc = nn.Linear(self.hidden_size, self.output_dim)
        if self.output_mode == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
        self.accuracy_function = init_accuracy_function(config, *args, **params)

    def init_multi_gpu(self, device, config, *args, **params) -> None:
        """Distribui o encoder BERT em múltiplas GPUs via ``nn.DataParallel``.

        Args:
            device: Lista de IDs de GPU passada ao ``DataParallel``.
            config: ConfigParser (não utilizado; mantido para interface uniforme).
        """
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    @staticmethod
    def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling sobre os token embeddings mascarados.

        Referência: Reimers & Gurevych (2019) — Sentence-BERT.
        Preferível ao pooler_output (tanh([CLS])) para tarefas de similaridade
        semântica, pois agrega toda a sequência com pesos uniformes.

        Args:
            last_hidden_state: Tensor [B, L, H] — saída da última camada do encoder.
            attention_mask:    Tensor [B, L]    — 1 para tokens reais, 0 para padding.
        Returns:
            Tensor [B, H] — embedding médio por amostra do batch.
        """
        # Expande a máscara para [B, L, H] e converte para float
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
        # clamp evita divisão por zero em sequências totalmente mascaradas
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, data: dict, config, gpu_list, acc_result, mode: str) -> dict:
        """Passagem forward: BERT → mean-pool → [FC → perda | pool-out].

        Args:
            data:       Dict com ``'input_ids'``, ``'attention_mask'``,
                        ``'token_type_ids'``, ``'guid'`` e, em treino/val, ``'label'``.
            config:     ConfigParser com ``output.pool_out``.
            gpu_list:   Lista de IDs de GPU (não utilizado diretamente).
            acc_result: Acumulador de acurácia.
            mode:       ``'train'``, ``'valid'`` ou ``'test'``.

        Returns:
            Dict com chaves dependentes do modo:

            - ``train``: ``{'loss', 'acc_result'}``
            - ``valid``: ``{'loss', 'acc_result', 'output'}``
            - ``test`` + ``pool_out=True``: ``{'output'}`` com embeddings ``[B, H]``
            - ``test`` padrão: ``{'output'}`` com logits
        """
        input_ids, attention_mask, token_type_ids = data['input_ids'], data['attention_mask'], data['token_type_ids']
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # Mean pooling: agrega last_hidden_state ponderado pela attention_mask — [B, H]
        y = self._mean_pooling(outputs.last_hidden_state, attention_mask)
        if mode == 'test' and config.getboolean('output', 'pool_out'):
            output = []
            y = y.cpu().detach().numpy().tolist()
            for i, guid in enumerate(data['guid']):
                output.append([guid, y[i]])
            return {"output": output}

        y = self.fc(y)
        y = y.view(y.size()[0], -1)

        if mode == 'valid':
            label = data["label"]
            loss = self.criterion(y, label.view(-1))
            acc_result = self.accuracy_function(y, label, config, acc_result)
            output = []
            y = y.cpu().detach().numpy().tolist()
            # import pdb; pdb.set_trace()
            for i, guid in enumerate(data['guid']):
                output.append([guid, label[i], y[i]])
            return {"loss": loss, "acc_result": acc_result, "output": output}

        elif mode == 'train':
            label = data["label"]
            loss = self.criterion(y, label.view(-1))
            acc_result = self.accuracy_function(y, label, config, acc_result)
            return {"loss": loss, "acc_result": acc_result}

        else:
            output = []
            y = y.cpu().detach().numpy().tolist()
            for i, guid in enumerate(data['guid']):
                output.append([guid, y[i]])
            return {"output": output}
    
