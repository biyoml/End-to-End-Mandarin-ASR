""" Define useful functions for data I/O.
"""
import os
import glob


def encode_fn(s_in):
    """
    A function for Pytorch-NLP tokenizer to encode sequences.

    Args:
        s_in (string): Sentence.

    Returns:
        s_out (list(string)): Words.
    """
    s_out = s_in.split()
    return s_out


def decode_fn(s_in):
    """
    A function for Pytorch-NLP tokenizer to decode sequences.

    Args:
        s_in (list(string)): Words.

    Returns:
        s_out (string): Sentence.
    """
    s_out = []
    for w in s_in:
        if w == '<s>':
            continue
        elif w=='</s>':
            break
        s_out.append(w)
    s_out = ' '.join(s_out)
    return s_out
