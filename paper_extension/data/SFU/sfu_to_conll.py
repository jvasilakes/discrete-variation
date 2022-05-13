import os
import argparse
import xml.etree.ElementTree as ET

from sklearn.model_selection import KFold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_dirs", type=str, nargs='+',
                        help="""Directories containing XML files.""")
    parser.add_argument("--outdir", type=str, required=True,
                        help="""Where to save the processed files.""")
    parser.add_argument("--random_state", type=int, default=0,
                        help="""Random state for the train_test_split.""")
    return parser.parse_args()


def main(args):
    all_sentences = []
    all_neg_labels = []
    all_spec_labels = []
    for corpus_dir in args.corpus_dirs:
        files = os.listdir(corpus_dir)
        for file in files:
            filepath = os.path.join(corpus_dir, file)
            if filepath.endswith(".orig"):
                continue
            sentences, neg_labs = read_xml(filepath, modality="negation")
            sentences, spec_labs = read_xml(filepath, modality="speculation")
            all_sentences.extend(sentences)
            all_neg_labels.extend(neg_labs)
            all_spec_labels.extend(spec_labs)

    os.makedirs(args.outdir)

    all_data = list(zip(all_sentences, all_neg_labels, all_spec_labels))
    kf = KFold(n_splits=5)
    for (split, (train_idxs, val_idxs)) in enumerate(kf.split(all_sentences)):
        trainset = [all_data[i] for i in train_idxs]
        dev_i = len(val_idxs) // 2
        devset = [all_data[i] for i in val_idxs[:dev_i]]
        testset = [all_data[i] for i in val_idxs[dev_i:]]

        split_outdir = os.path.join(args.outdir, f"split_{split}")
        os.makedirs(split_outdir)
        train_file = os.path.join(split_outdir, "train.conll")
        format_and_save(trainset, train_file)
        dev_file = os.path.join(split_outdir, "dev.conll")
        format_and_save(devset, dev_file)
        test_file = os.path.join(split_outdir, "test.conll")
        format_and_save(testset, test_file)

    # train, evals = train_test_split(all_data, test_size=0.3,
    #                                 random_state=args.random_state)
    # dev, test = train_test_split(evals, test_size=0.5,
    #                              random_state=args.random_state)


def format_and_save(data, filepath):
    with open(filepath, 'w') as outF:
        for (sent, neg_labs, spec_labs) in data:
            for (tok, nl, sl) in zip(sent, neg_labs, spec_labs):
                outF.write(f"{tok}\t{nl}\t{sl}\n")
            outF.write('\n')


def read_xml(file, modality="negation"):
    NULL_LABEL = 'O'

    if modality not in ["negation", "speculation"]:
        raise ValueError(f"Unsupported modality '{modality}' (negation or speculation)")  # noqa
    attr = modality[0].upper()  # N or S

    try:
        tree = ET.parse(file)
    except ET.ParseError:
        print(file)
        print("not well formed. continuing...")
        return []
    root = tree.getroot()

    out_sentences = []
    out_labels = []
    for sentence in root.findall(".//SENTENCE"):
        elements = sentence.findall("./*")

        processed_sentence = []  # list of tokens
        labels = []  # list of labels, one per token
        cue_ids = {}
        cue_count = 0
        for elem in elements:
            # Unscoped tokens.
            if elem.tag == 'W':
                if not elem.text.strip():
                    continue
                processed_sentence.append(elem.text)
                labels.append(NULL_LABEL)

            # Cue tokens
            elif elem.tag in ['C', "cue"]:
                if elem.tag == 'C':
                    cue = elem.find("./cue")
                else:
                    cue = elem
                lab = NULL_LABEL
                if cue is not None:
                    mod = cue.get("type")
                    if mod == modality:
                        uid = cue.get("ID")
                        normed_id = len(cue_ids)
                        cue_ids[uid] = normed_id
                        lab = f"{attr}_cue_{normed_id}"
                        cue_count += 1
                    else:
                        lab = NULL_LABEL
                cue_tokens = elem.findall(".//W")
                for tok in cue_tokens:
                    processed_sentence.append(tok.text)
                    labels.append(lab)

            # Scoped tokens
            elif elem.tag == "xcope":
                src_id = elem.find("./ref").get("SRC")
                scoped_tokens = elem.findall(".//W")
                if src_id in cue_ids.keys():
                    normed_id = cue_ids[src_id]
                    lab = f"{attr}_{normed_id}"
                else:
                    lab = NULL_LABEL
                for tok in scoped_tokens:
                    processed_sentence.append(tok.text)
                    labels.append(lab)

        out_sentences.append(processed_sentence)
        out_labels.append(labels)
    return out_sentences, out_labels


if __name__ == "__main__":
    args = parse_args()
    main(args)
