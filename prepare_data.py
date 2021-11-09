""" List audio files and transcripts, then create Pytorch-NLP tokenizer.
"""
import torch
import os
import argparse
import glob
import pandas as pd
import data_utils
from torchnlp.encoders.text import StaticTokenizerEncoder


def read_transcripts(root):
    """
    Returns:
        transcripts (dict): All the transcripts from AISHELL dataset. They are represented
                            by {audio id: transcript}.
    """
    with open(os.path.join(root, "transcript/aishell_transcript_v0.8.txt")) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    transcripts = {}
    for l in lines:
        l = l.split()
        id = l[0]
        seq = ''.join(l[1:])
        seq = ' '.join(list(seq))
        transcripts[id] = seq
    return transcripts


def get_id(audio_file):
    """
    Given an audio file path, return its ID.
    """
    return os.path.basename(audio_file)[:-4]


def process_dataset(root, split):
    """
    List audio files and transcripts for a certain partition of AISHELL.

    Args:
        root (string): The root directory of AISHELL dataset.
        split (string): Which of the subset of data to take. One of 'train', 'dev' or 'test'.
    """
    transcripts = read_transcripts(root)

    # Collect audios belonging to this split.
    audio_files = glob.glob(os.path.join(root, "wav/%s/*/*.wav" % split))
    # Ignore audios without transcript.
    audio_files = [a for a in audio_files if get_id(a) in transcripts]
    # Collect transcripts belonging to this split.
    transcripts = [transcripts[get_id(a)] for a in audio_files]

    fname = '%s.csv'%split.upper()
    with open(fname, 'w') as f:
        f.write("audio,transcript\n")
        count = 0
        for (x, y) in zip(audio_files, transcripts):
            f.write("%s,%s\n" % (x, y))
            count += 1
    print ("%s is created. (%d examples)" % (fname, count))


def create_tokenizer():
    """
    Create and save Pytorch-NLP tokenizer.

    Args:
        root (string): The root directory of AISHELL dataset.
    """
    transcripts = pd.read_csv('TRAIN.csv')['transcript']
    tokenizer = StaticTokenizerEncoder(transcripts,
                                       append_sos=True,
                                       append_eos=True,
                                       tokenize=data_utils.encode_fn,
                                       detokenize=data_utils.decode_fn)
    torch.save(tokenizer, 'tokenizer.pth')


def main():
    parser = argparse.ArgumentParser(description="Make lists of audio files and transcripts, and create tokenizer.")
    parser.add_argument('root', type=str, help="The root directory of AISHELL dataset.")
    args = parser.parse_args()

    process_dataset(args.root, 'train')
    process_dataset(args.root, 'dev')
    process_dataset(args.root, 'test')

    create_tokenizer()
    print ("Data preparation is complete!")


if __name__ == '__main__':
    main()
