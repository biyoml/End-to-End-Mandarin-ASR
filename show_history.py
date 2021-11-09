""" Visualize training history.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Visualize training history.")
    parser.add_argument('history', type=str, help="Path to the history file.")
    args = parser.parse_args()

    df = pd.read_csv(args.history)
    # header: datetime, epoch, learning rate, train loss, dev loss, error rate

    plt.figure(figsize=(15,3))
    plt.subplots_adjust(.05, 0.15, .95, .9, None, None)

    plt.subplot(1,3,1)
    plt.title("Loss")
    plt.plot(df['epoch'], df['train loss'], label='train')
    plt.plot(df['epoch'], df['dev loss'], label='dev')
    plt.xlabel('epochs')
    plt.legend()

    plt.subplot(1,3,2)
    plt.title("Dev. error rate")
    plt.grid()
    plt.plot(df['epoch'], df['error rate'])
    plt.ylim(0,1)
    plt.xlabel('epochs')

    plt.subplot(1,3,3)
    plt.title("Learning rate")
    plt.plot(df['epoch'], df['learning rate'])
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel('epochs')

    plt.show()


if __name__ == '__main__':
    main()