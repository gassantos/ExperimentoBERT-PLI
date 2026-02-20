import torch.optim as optim
from transformers import AdamW


def init_optimizer(model, config, *args, **params):
    optimizer_type = config.get("train", "optimizer")
    learning_rate = config.getfloat("train", "learning_rate")
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "bert_adam":
        # AdamW do transformers é o sucessor moderno do BertAdam.
        # O warmup linear é tratado por get_linear_schedule_with_warmup em train_tool.py.
        optimizer = AdamW(model.parameters(), lr=learning_rate,
                          weight_decay=config.getfloat("train", "weight_decay"))
    else:
        raise NotImplementedError

    return optimizer
