"""model/loss.py — Funções e módulos de perda para classificação
===================================================================
Disponibiliza três estratégias de perda usadas no pipeline BERT-PLI:

- :class:`MultiLabelSoftmaxLoss` — cross-entropy ponderada por tarefa para classificação multi-label.
- :func:`multi_label_cross_entropy_loss` — perda binária element-wise (BCE sem sigmoid).
- :func:`cross_entropy_loss` — cross-entropy padrão do PyTorch.
- :class:`FocalLoss` — perda focal (Lin et al., 2017) para dados desbalanceados.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelSoftmaxLoss(nn.Module):
    """Cross-entropy ponderada por tarefa para classificação multi-label.

    Cada tarefa pode ter um peso de classe configurado via ``loss_weight_<i>``
    na seção ``[train]`` do arquivo ``.config``. O tensor de peso é criado
    no forward no device do input, garantindo compatibilidade com CPU, CUDA e MPS.
    """

    def __init__(self, config):
        """Args:
            config: ConfigParser com ``model.output_dim`` e, opcionalmente,
                ``train.loss_weight_<i>`` para cada tarefa.
        """
        super(MultiLabelSoftmaxLoss, self).__init__()
        self.task_num = config.getint("model", "output_dim")
        # Armazena os pesos como floats Python — o tensor é criado no forward()
        # com o device correto, eliminando o .cuda() hardcoded que quebrava
        # em CPU e Apple MPS.
        self._loss_weights = []
        for a in range(self.task_num):
            try:
                ratio = config.getfloat("train", "loss_weight_%d" % a)
                self._loss_weights.append(ratio)
            except Exception:
                self._loss_weights.append(None)

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calcula a perda somada de todas as tarefas.

        Args:
            outputs: Tensor ``[B, num_tasks, num_classes]`` com logits.
            labels:  Tensor ``[B, num_tasks]`` com rótulos inteiros.

        Returns:
            Escalar de perda somada sobre todas as tarefas.
        """
        loss = 0
        for a in range(0, len(outputs[0])):
            o = outputs[:, a, :].view(outputs.size()[0], -1)
            w = self._loss_weights[a]
            if w is not None:
                # Tensor criado no device do input — funciona em CPU, CUDA e MPS
                weight = torch.tensor([1.0, w], dtype=torch.float32, device=o.device)
                criterion = nn.CrossEntropyLoss(weight=weight)
            else:
                criterion = nn.CrossEntropyLoss()
            loss += criterion(o, labels[:, a])
        return loss


def multi_label_cross_entropy_loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Perda binária element-wise para classificação multi-label.

    Computa ``-y*log(p) - (1-y)*log(1-p)`` por elemento e retorna a média
    sobre o batch. Assume que ``outputs`` já passou por sigmoid.

    Args:
        outputs: Tensor ``[B, C]`` com probabilidades em ``(0, 1)``.
        labels:  Tensor ``[B, C]`` com rótulos binários.

    Returns:
        Escalar de perda média.
    """
    labels = labels.float()
    temp = outputs
    res = - labels * torch.log(temp) - (1 - labels) * torch.log(1 - temp)
    res = torch.mean(torch.sum(res, dim=1))

    return res


def cross_entropy_loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Wrapper conveniente em torno de ``nn.CrossEntropyLoss``.

    Args:
        outputs: Tensor ``[B, C]`` com logits.
        labels:  Tensor ``[B]`` com rótulos inteiros.

    Returns:
        Escalar de perda cross-entropy.
    """
    criterion = nn.CrossEntropyLoss()
    return criterion(outputs, labels)


class FocalLoss(nn.Module):
    """Perda focal para lidar com desbalanceamento de classes.

    Reduz a contribuição de exemplos fáceis (alta confiança) e foca o
    treino em exemplos difíceis via fator ``(1 - p_t)^gamma``.

    Referência: Lin et al. (2017) — *Focal Loss for Dense Object Detection*.
    """

    def __init__(self, gamma: float = 0, alpha: torch.Tensor | None = None, size_average: bool = True):
        """Args:
            gamma: Fator de foco. ``0`` equivale a cross-entropy padrão.
            alpha: Tensor de pesos por classe. ``None`` = sem pesos.
            size_average: Se ``True``, retorna média; caso contrário, soma.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calcula a perda focal.

        Args:
            input:  Tensor ``[B, C]`` (ou ``[B, C, H, W]``) com logits.
            target: Tensor ``[B]`` (ou ``[B, H, W]``) com rótulos inteiros.

        Returns:
            Escalar de perda focal (média ou soma conforme ``size_average``).
        """
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.detach().exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
