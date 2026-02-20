import torch.optim as optim


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
        # torch.optim.AdamW é o sucessor moderno do BertAdam (transformers.AdamW removido na v4.46+).
        # O warmup linear é tratado por get_linear_schedule_with_warmup em train_tool.py.
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                weight_decay=config.getfloat("train", "weight_decay"))
    else:
        raise NotImplementedError

    return optimizer
