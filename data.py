""" Load and preprocess data.
"""
import torch
import torchaudio
import os
import pandas as pd
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader


class ASR(Dataset):
    """
    Stores a Pandas DataFrame in __init__, and reads and preprocesses examples in __getitem__.
    """
    def __init__(self, split, augmentation):
        """
        Args:
            augmentation (bool): Apply SpecAugment to training data or not.
        """
        self.df = pd.read_csv('%s.csv' % split.upper())
        self.tokenizer = torch.load('tokenizer.pth')
        self.augmentation = (augmentation and (split.upper() == 'TRAIN'))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            x (torch.FloatTensor, [seq_length, dim_features]): The FBANK features.
            y (torch.LongTensor, [n_tokens]): The label sequence.
        """
        x, y = self.df.iloc[idx]
        x, sample_rate = torchaudio.load(x)
        # Compute filter bank features
        x = torchaudio.compliance.kaldi.fbank(x, num_mel_bins=80, sample_frequency=sample_rate)   # [n_windows, 80]
        # CMVN
        x = self.cmvn(x)
        # SpecAugment
        if self.augmentation:
            x = self.specaugment(x)
        # Stack every 3 frames and down-sample frame rate by 3, following https://arxiv.org/abs/1712.01769.
        x = x[:(x.shape[0]//3)*3].view(-1,3*80)   # [n_windows, 80] --> [n_windows//3, 240]
        # Tokenization
        y = self.tokenizer.encode(y)
        return x, y

    def cmvn(self, x):
        """
        Cepstral mean and variance normalization.
        """
        mean = torch.mean(x, dim=0)   # [80]
        x = x - mean                  # [n_windows, 80]
        std = torch.std(x, dim=0)     # [80]
        x = x / (std + 1e-10)         # [n_windows, 80]
        return x

    def specaugment(self, x, F=15, mF=2, T=70, p=0.2, mT=2):
        # TODO: Allow user to tune these parameters in config file.
        """
        SpecAugment (https://arxiv.org/abs/1904.08779). We discard the time warping policy for simplicity.

        Args:
            x (torch.FloatTensor, [seq_length, dim_features]): The FBANK features.
            F, mF, T, p, mT: The parameters referred in SpecAugment paper.
        """
        x = x.T   # [n_windows, 80] --> [80, n_windows]

        # Freq. masking
        for _ in range(mF):
            x = torchaudio.transforms.FrequencyMasking(F)(x)

        # Time masking
        Tclamp = min(T, int(p * x.shape[1]))
        for _ in range(mT):
            x = torchaudio.transforms.TimeMasking(Tclamp)(x)
        return x.T

    def generateBatch(self, batch):
        """
        Generate a mini-batch of data. For DataLoader's 'collate_fn'.

        Args:
            batch (list(tuple)): A mini-batch of (FBANK features, label sequences) pairs.

        Returns:
            xs (torch.FloatTensor, [batch_size, (padded) seq_length, dim_features]): A mini-batch of FBANK features.
            xlens (torch.LongTensor, [batch_size]): Sequence lengths before padding.
            ys (torch.LongTensor, [batch_size, (padded) n_tokens]): A mini-batch of label sequences.
        """
        xs, ys = zip(*batch)
        xlens = torch.tensor([x.shape[0] for x in xs])
        xs = rnn_utils.pad_sequence(xs, batch_first=True)   # [batch_size, (padded) seq_length, dim_features]
        ys = rnn_utils.pad_sequence(ys, batch_first=True)   # [batch_size, (padded) n_tokens]
        return xs, xlens, ys


def load(split, batch_size, workers=0, augmentation=False):
    """
    Args:
        split (string): Which of the subset of data to take. One of 'train', 'dev' or 'test'.
        batch_size (integer): Batch size.
        workers (integer): How many subprocesses to use for data loading.
        augmentation (bool): Apply SpecAugment to training data or not.

    Returns:
        loader (DataLoader): A DataLoader can generate batches of (FBANK features, FBANK lengths, label sequence).
    """
    assert split in ['train', 'dev', 'test']

    dataset = ASR(split, augmentation)
    print ("%s set size:"%split.upper(), len(dataset))
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        collate_fn=dataset.generateBatch,
                        shuffle=True,
                        num_workers=workers,
                        pin_memory=True)
    return loader


def inspect_data():
    """
    Test the functionality of input pipeline and visualize a few samples.
    """
    import matplotlib.pyplot as plt

    BATCH_SIZE = 64
    SPLIT = 'train'

    loader = load(SPLIT, BATCH_SIZE)
    tokenizer = torch.load('tokenizer.pth')
    print ("Vocabulary size:", len(tokenizer.vocab))
    print (tokenizer.vocab)

    xs, xlens, ys = next(iter(loader))
    print (xs.shape, ys.shape)
    for i in range(BATCH_SIZE):
        print (ys[i])
        print (tokenizer.decode(ys[i]))
        plt.figure()
        plt.imshow(xs[i].T)
        plt.show()


if __name__ == '__main__':
    inspect_data()
