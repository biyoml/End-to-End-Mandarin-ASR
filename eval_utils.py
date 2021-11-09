""" Define necessary functions for evaluation.
"""
import torch
import editdistance
import numpy as np
from tqdm import tqdm


def eval_dataset(dataloader, model, beam_width=1):
    """
    Calculate loss and error rate on a dataset.
    """
    tokenizer = torch.load('tokenizer.pth')
    total_loss = []
    n_tokens = 0
    total_error = 0
    with torch.no_grad():
        eval_tqdm = tqdm(dataloader, desc="Evaluating")
        for (xs, xlens, ys) in eval_tqdm:
            total_loss.append(model(xs.cuda(), xlens, ys.cuda()).item())
            preds_batch, _ = model(xs.cuda(), xlens, beam_width=beam_width)   # [batch_size, 100]
            for i in range(preds_batch.shape[0]):
                preds = tokenizer.decode(preds_batch[i])
                gt = tokenizer.decode(ys[i])
                preds = preds.split()
                gt = gt.split()
                total_error += editdistance.eval(gt, preds)
                n_tokens += len(gt)
            # Show message
            loss = np.mean(total_loss)
            error = total_error / n_tokens
            eval_tqdm.set_postfix(loss="%.3f"%loss, error="%.4f"%error)
    return loss, error
