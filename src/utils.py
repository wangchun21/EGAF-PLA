import torch
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import LambdaLR

def set_seed(seed=123):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, max_epochs):
    """Cosine annealing scheduler with warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
    return LambdaLR(optimizer, lr_lambda)

def c_index(y_true, y_pred):
    summ = 0
    pair = 0
    for i in range(1, len(y_true)):
        for j in range(i):
            pair += 1
            if y_true[i] > y_true[j]:
                summ += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            elif y_true[i] < y_true[j]:
                summ += 1 * (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            else:
                pair -= 1
    return summ / pair if pair != 0 else 0

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def CORR(y_true, y_pred):
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    return pearsonr(y_true, y_pred)[0]

def SD(y_true, y_pred):
    from sklearn.linear_model import LinearRegression
    y_pred = y_pred.reshape((-1, 1))
    lr = LinearRegression().fit(y_pred, y_true)
    y_ = lr.predict(y_pred)
    return np.sqrt(np.square(y_true - y_).sum() / (len(y_pred) - 1))