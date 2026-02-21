# -*- coding: utf-8 -*-
__author__ = 'yshao'

import torch
import torch.nn as nn
from transformers import BertModel

from tools.accuracy_init import init_accuracy_function


class BertPoint(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
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

    def init_multi_gpu(self, device, config, *args, **params):
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

    def forward(self, data, config, gpu_list, acc_result, mode):
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
    
