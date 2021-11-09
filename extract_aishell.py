""" Extract data_aishell.tgz.
"""
import os
import tarfile
import glob
import argparse


def extracttar(filename, outpath="./"):
    with tarfile.open(filename) as f:
        for item in f:
            print ("Extract %s" % item.name)
            f.extract(item, path=outpath)

def main():
    parser = argparse.ArgumentParser(description="Extract data_aishell.tgz.")
    parser.add_argument('file', type=str, help="Path to the data_aishell.tgz.")
    args = parser.parse_args()

    dirname = os.path.dirname(args.file)
    extracttar(args.file, outpath=dirname)
    dirname = os.path.join(dirname, 'data_aishell/wav')
    files = glob.glob(os.path.join(dirname, "*.gz"))
    for f in files:
        extracttar(f, outpath=dirname)
    for f in files:
        os.remove(f)
    print ("Completed !")


if __name__ == '__main__':
    main()