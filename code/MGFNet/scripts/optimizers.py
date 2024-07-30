from torch import nn
from torch.optim import AdamW, SGD, Adagrad, RMSprop, Adadelta, Adam


def get_optimizer(model: nn.Module, optimizer: str, lr: float, weight_decay: float = 0.01):
    wd_params, nwd_params = [], []
    for p in model.parameters():
        if p.dim() == 1:
            nwd_params.append(p)
        else:
            wd_params.append(p)

    params = [
        {"params": wd_params},
        {"params": nwd_params, "weight_decay": 0}
    ]

    if optimizer.lower() == 'adamw':
        return AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    elif optimizer.lower() == 'sgd':
        return SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer.lower() == 'adagrad':
        return Adagrad(params, lr, weight_decay=weight_decay)
    elif optimizer.lower() == 'rmsprop':
        return RMSprop(params, lr, alpha=0.9, eps=1e-8, weight_decay=weight_decay)
    elif optimizer.lower() == 'adadelta':
        return Adadelta(params, lr, rho=0.9, eps=1e-8, weight_decay=weight_decay)
    elif optimizer.lower() == 'adam':
        return Adam(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    else:
        raise ValueError(
            f"Optimizer '{optimizer}' not supported. Choose from 'adamw', 'sgd', 'adagrad', 'rmsprop', 'adadelta', or 'adam'.")
