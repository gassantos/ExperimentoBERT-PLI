# -*- coding: utf-8 -*-
"""model/nlp/AttenRNN.py — Agregação de interações padrão com RNN + Atenção
==========================================================================
Estágio 4 do pipeline BERT-PLI: recebe a matriz de interações padrão-parágrafo
gerada pelo :class:`BertPoolOutMax` e produz um score binário de entailment.

Arquitetura::

    entrada [B, M, H] → LSTM/GRU → MaxPool temporal → Atenção → FC → logits [B, 2]

Onde:
  - B = batch size
  - M = número máximo de parágrafos da query (``max_para_q``)
  - H = ``bert_hidden_size`` (padrão 768)
"""
__author__ = 'yshao'

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.accuracy_init import init_accuracy_function


class Attention(nn.Module):
    """Módulo de atenção global sobre a saída da RNN.

    Computa um score de atenção escalar por passo temporal via produto interno
    entre cada hidden state e o vetor de feature global (max-pool), produzindo
    um único vetor de contexto pela soma ponderada.
    """

    def __init__(self, config):
        """Args:
            config: ConfigParser (reservado para extensões futuras; não utilizado atualmente).
        """
        super(Attention, self).__init__()
        pass

    def forward(self, feature: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Aplica atenção dot-product sobre a sequência RNN.

        Args:
            feature: Tensor ``[B, H, 1]`` — vetor de feature global (max-pool).
            hidden:  Tensor ``[B, M, H]`` — saída da RNN por passo temporal.

        Returns:
            Tensor ``[B, H]`` — vetor de contexto ponderado pela atenção.
        """
        # hidden: B * M * H, feature: B * H * 1
        ratio = torch.bmm(hidden, feature)
        # ratio: B * M * 1
        ratio = ratio.view(ratio.size(0), ratio.size(1))
        ratio = F.softmax(ratio, dim=1).unsqueeze(2)
        # result: B * H
        result = torch.bmm(hidden.permute(0, 2, 1), ratio)
        result = result.view(result.size(0), -1)
        return result


class AttentionRNN(nn.Module):
    """Classificador de entailment baseado em RNN bidirecional com atenção.

    Recebe uma matriz de interações ``[B, M, H]`` gerada no estágio poolout e
    aplica uma LSTM ou GRU bidirecional seguida de max-pooling temporal,
    atenção e camada linear de classificação binária.

    Parâmetros lidos do ``.config`` (seção ``[model]``):
        ``rnn`` (``'lstm'`` ou ``'gru'``), ``hidden_dim``, ``num_layers``,
        ``bidirectional``, ``dropout_rnn``, ``dropout_fc``, ``max_para_q``,
        ``output_dim``, ``bert_hidden_size`` (padrão 768).
    """

    def __init__(self, config, gpu_list, *args, **params):
        """Inicializa RNN, atenção, camadas lineares e funções de perda/acurácia.

        Args:
            config:   ConfigParser com parâmetros do modelo e treino.
            gpu_list: Lista de IDs de GPU disponíveis (usada em ``init_weight``).
        """
        super(AttentionRNN, self).__init__()

        # Dimensão de entrada = hidden_size do backbone BERT que gerou os embeddings.
        # Lida do config para desacoplar AttenRNN do backbone concreto.
        # Fallback 768 garante retrocompatibilidade com configs existentes.
        self.input_dim = config.getint('model', 'bert_hidden_size', fallback=768)
        self.hidden_dim = config.getint('model', 'hidden_dim')
        self.dropout_rnn = config.getfloat('model', 'dropout_rnn')
        self.dropout_fc = config.getfloat('model', 'dropout_fc')
        self.bidirectional = config.getboolean('model', 'bidirectional')
        if self.bidirectional:
            self.direction = 2
        else:
            self.direction = 1
        self.num_layers = config.getint("model", 'num_layers')
        self.output_dim = config.getint("model", "output_dim")
        self.max_para_q = config.getint('model', 'max_para_q')

        if config.get('model', 'rnn') == 'lstm':
            self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout_rnn)
        else:
            self.rnn = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout_rnn)

        self.max_pool = nn.MaxPool1d(kernel_size=self.max_para_q)
        self.fc_a = nn.Linear(self.hidden_dim*self.direction, self.hidden_dim*self.direction)
        self.attention = Attention(config)
        self.fc_f = nn.Linear(self.hidden_dim*self.direction, self.output_dim)
#         self.soft_max = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(self.dropout_fc)
        self.weight = self.init_weight(config, gpu_list)
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)
        self.accuracy_function = init_accuracy_function(config, *args, **params)

    def init_weight(self, config, gpu_list) -> torch.Tensor | None:
        """Cria tensor de pesos de classe para a cross-entropy, se configurado.

        Args:
            config:   ConfigParser com a chave opcional ``model.label_weight``.
            gpu_list: Lista de IDs de GPU; o tensor é movido para CUDA se disponível.

        Returns:
            Tensor de pesos ``[output_dim]`` ou ``None`` se ``label_weight`` não estiver definido.
        """
        try:
            label_weight = config.getfloat('model', 'label_weight')
        except Exception:
            return None
        weight_lst = torch.ones(self.output_dim)
        weight_lst[-1] = label_weight
        if torch.cuda.is_available() and len(gpu_list) > 0:
            weight_lst = weight_lst.cuda()
        return weight_lst

    def init_hidden(self, config, batch_size: int, gpu_list) -> None:
        """Inicializa o estado oculto da RNN com zeros no device correto.

        Para LSTM, cria uma tupla ``(h_0, c_0)``; para GRU, cria apenas ``h_0``.

        Args:
            config:     ConfigParser para determinar o tipo de RNN (``model.rnn``).
            batch_size: Tamanho do batch atual.
            gpu_list:   Lista de IDs de GPU (define CPU vs CUDA).
        """
        device = torch.device("cuda" if torch.cuda.is_available() and len(gpu_list) > 0 else "cpu")
        shape = (self.direction * self.num_layers, batch_size, self.hidden_dim)
        if config.get('model', 'rnn') == 'lstm':
            self.hidden = (
                torch.zeros(shape, device=device),
                torch.zeros(shape, device=device),
            )
        else:
            self.hidden = torch.zeros(shape, device=device)

    def init_multi_gpu(self, device, config, *args, **params) -> None:
        """Distribui os sub-módulos em múltiplas GPUs via ``nn.DataParallel``.

        Args:
            device: Lista de IDs de GPU passada ao ``DataParallel``.
            config: ConfigParser (não utilizado; mantido para interface uniforme).
        """
        self.rnn = nn.DataParallel(self.rnn, device_ids=device)
        self.max_pool = nn.DataParallel(self.max_pool, device_ids=device)
        self.fc_a = nn.DataParallel(self.fc_a, device_ids=device)
        self.attention = nn.DataParallel(self.attention, device_ids=device)
        self.fc_f = nn.DataParallel(self.fc_f, device_ids=device)
#         self.soft_max = nn.DataParallel(self.soft_max, device_ids=device)

    def forward(self, data: dict, config, gpu_list, acc_result, mode: str) -> dict:
        """Passagem forward completa: RNN → MaxPool → Atenção → FC → perda/output.

        Args:
            data:       Dict com ``'input'`` ``[B, M, H]`` e, em modo ``train``/``valid``,
                        ``'label'`` ``[B]`` e ``'guid'`` (lista de IDs de exemplo).
            config:     ConfigParser com parâmetros de execução.
            gpu_list:   Lista de IDs de GPU (repassada a ``init_hidden``).
            acc_result: Acumulador de acurácia.
            mode:       ``'train'``, ``'valid'`` ou ``'test'``.

        Returns:
            Dict com chaves dependentes do modo:

            - ``train``: ``{'loss', 'acc_result'}``
            - ``valid``: ``{'loss', 'acc_result', 'output'}``
            - ``test``/padrão: ``{'output'}``
        """
        x = data['input'] # B * M * I
        batch_size = x.size()[0]
        self.init_hidden(config, batch_size, gpu_list) # 2 * B * H

        rnn_out, self.hidden = self.rnn(x, self.hidden) # rnn_out: B * M * 2H, hidden: 2 * B * H
        tmp_rnn = rnn_out.permute(0, 2, 1) # B * 2H * M

        feature = self.max_pool(tmp_rnn) # B * 2H * 1
        feature = feature.squeeze(2) # B * 2H
        feature = self.fc_a(feature) # B * 2H
        feature = feature.unsqueeze(2) # B * 2H * 1

        atten_out = self.attention(feature, rnn_out) # B * (2H)
        atten_out = self.dropout(atten_out)
        y = self.fc_f(atten_out)
#         y = self.soft_max(y)
        y = y.view(y.size()[0], -1)

        if 'label' in data.keys():
            label = data['label']
            loss = self.criterion(y, label.view(-1))
            acc_result = self.accuracy_function(y, label, config, acc_result)
            if mode == 'valid':
                output = []
                y = y.cpu().detach().numpy().tolist()
                for i, guid in enumerate(data['guid']):
                    output.append([guid, label[i], y[i]])
                return {"loss": loss, "acc_result": acc_result, "output": output}
            elif mode == 'test':
                output = []
                y = y.cpu().detach().numpy().tolist()
                for i, guid in enumerate(data['guid']):
                    output.append([guid, y[i]])
                return {"output": output}
            return {"loss": loss, "acc_result": acc_result}
        else:
            output = []
            y = y.cpu().detach().numpy().tolist()
            for i, guid in enumerate(data['guid']):
                output.append([guid, y[i]])
            return {"output": output}










