import gc
import torch

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

def get_accuracy(preds, y):
    preds = preds.argmax(1)
    num_correct = (preds == y).sum().item()
    acc = num_correct / y.shape[0]
    return acc