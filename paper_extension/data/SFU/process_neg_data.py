import os
import re
import pickle
import argparse
import logging
from glob import glob

from sklearn.model_selection import train_test_split
from collections import Counter


"""
Meant for processing CONLL formatted negation scope data.
See data/SFU/processed/
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_splits_dir", type=str, required=True,
                        help="Path to directory containing data splits.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Where to save the processed data.")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of labeled to unlabeled data.")
    args = parser.parse_args()
    return args


def process(word):
    word = "".join(c if not c.isdigit() else '0' for c in word)
    return word


def process_file(data_file):
    logging.info("loading data from " + data_file + " ...")
    sents = []
    tags = []
    sent = []
    tag = []
    with open(data_file, 'r', encoding='utf-8') as df:
        for line in df:
            if line[0:10] == '-DOCSTART-':
                continue
            if line.strip():
                # ignore the speculation label
                word, t, _ = line.strip().split('\t')
                sent.append(process(word))
                t = re.sub(r'_[0-9]+', '', t)
                tag.append(t)  # remove the cue number for now
            else:
                if sent and tag:
                    sents.append(sent)
                    tags.append(tag)
                sent = []
                tag = []
    return sents, tags


def get_unlabeled_data(trainset, ratio, outfile):
    n_unlabel = len(trainset[0]) // 2
    X_train, X_test, y_train, y_test = train_test_split(
        trainset[0], trainset[1], test_size=args.ratio)
    other = [X_train, y_train]
    train = [X_test, y_test]

    X_train, X_test, y_train, y_test = \
        train_test_split(other[0], other[1], test_size=n_unlabel)

    unlabel_data = X_test
    logging.info(f"  #unlabeled data: {len(X_test)}")

    with open(outfile, "w+", encoding='utf-8') as fp:
        fp.write("\n".join([" ".join([w for w in sent])
                            for sent in unlabel_data]))
    logging.info(f"  unlabeled data saved to {outfile}")
    return train


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M')
    args = get_args()
    os.makedirs(args.outdir)

    # Individual split directories containing {train,dev,test}.conll
    split_dirs = glob(os.path.join(args.data_splits_dir, "split_*"))

    seen_tags = None
    for split_dir in split_dirs:
        split_num = split_dir.split('_')[-1]
        print(f"Processing split {split_num}")
        train = process_file(os.path.join(split_dir, "train.conll"))
        dev = process_file(os.path.join(split_dir, "dev.conll"))
        test = process_file(os.path.join(split_dir, "test.conll"))

        # sum(train[1], []) flattens the list of tags.
        tag_counter = Counter(sum(train[1], []) +
                              sum(dev[1], []) + sum(test[1], []))
        if seen_tags is None:
            seen_tags = set(tag_counter.keys())
            tagfile = os.path.join(args.outdir, "tagfile")
            with open(tagfile, "w+", encoding='utf-8') as fp:
                fp.write('\n'.join(sorted(tag_counter.keys())))
        else:
            if set(tag_counter.keys()) != seen_tags:
                raise ValueError("Splits contain different sets of tags!")

        if args.ratio < 1.0:
            unlabeled_outfile = os.path.join(
                args.outdir, f"unlabeled_ratio{args.ratio}_split{split_num}")
            train = get_unlabeled_data(train, args.ratio, unlabeled_outfile)

        logging.info("  #train data: {}".format(len(train[0])))
        logging.info("  #dev data: {}".format(len(dev[0])))
        logging.info("  #test data: {}".format(len(test[0])))

        outfile = os.path.join(
            args.outdir, f"labeled_ratio{args.ratio}_split{split_num}.data")
        pickle.dump([train, dev, test], open(outfile, 'wb'), protocol=-1)
