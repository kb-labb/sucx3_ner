# TODO
# split function that splits evenly not just completely randomly
# give each sentence tag count information per tag
# calculate 64 - 16 - 20 split for NE sentences
# split such that tags are evenly distributed
# fill up with the non-NE sentences

import json
import random
import argparse
from typing import Tuple, Dict, Iterable, Set, List
from compare_data import Sentence, read_json, write_data
from pprint import pprint
from collections import Counter


def split_ne_sentences(sentences: Iterable[Sentence]
                       ) -> Tuple[List[Sentence], List[Sentence]]:
    no_ne_sentences = []
    ne_sentences = []
    for sen in sentences:
        if any(t != "O" for t in sen["ner_tags"]):
            ne_sentences.append(sen)
        else:
            no_ne_sentences.append(sen)
    return no_ne_sentences, ne_sentences


def split_sort(sentences: Iterable[Sentence], out_folder: str) -> None:
    sentences = list(sentences)
    total, has_ne, ne_dict = ne_counts(sentences)
    ts = get_sizes(total)
    ns = get_sizes(has_ne)
    pprint([(k, get_sizes(v)) for k, v in sorted(ne_dict.items(), key=lambda x: x[1]) if k.startswith("B")])
    train: List[Sentence] = []
    dev: List[Sentence] = []
    test: List[Sentence] = []
    max_train, max_dev, max_test = get_sizes(has_ne)
    split_dict_train = {}
    split_dict_dev = {}
    split_dict_test = {}
    for k, v in filter(lambda x: x[0].startswith("B"), ne_dict.items()):
        vs = get_sizes(v)
        split_dict_train[k] = vs[0]
        split_dict_dev[k] = vs[1]
        split_dict_test[k] = vs[2]
    no_ne_sentences, ne_sentences = split_ne_sentences(sentences)
    random.shuffle(ne_sentences)
    random.shuffle(no_ne_sentences)

    def fits_in_split(size: int,
                      max_size: int,
                      split_dict: Dict[str, int],
                      tag_cnts: Dict[str, int],
                      ) -> bool:
        if size < max_size:
            if all(split_dict[t] - tag_cnts[t] >= 0 for t in tag_cnts):
                return True
        return False

    for sentence in ne_sentences:
        cnt = Counter(t for t in sentence["ner_tags"] if t.startswith("B"))
        if fits_in_split(len(test), max_test, split_dict_test, cnt):
            for k in cnt:
                split_dict_test[k] -= cnt[k]
            test.append(sentence)
        elif fits_in_split(len(dev), max_dev, split_dict_dev, cnt):
            for k in cnt:
                split_dict_dev[k] -= cnt[k]
            dev.append(sentence)
        else:
            for k in cnt:
                split_dict_train[k] -= cnt[k]
            train.append(sentence)
    pprint(split_dict_test)
    pprint(split_dict_dev)
    pprint(split_dict_train)

    max_train, max_dev, max_test = get_sizes(total)
    for sentence in no_ne_sentences:
        if len(test) < max_test:
            test.append(sentence)
        elif len(dev) < max_dev:
            dev.append(sentence)
        else:
            train.append(sentence)

    print("Train:")
    print(ts[0], ns[0])
    pprint(ne_counts(train))
    print("Development:")
    print(ts[1], ns[1])
    pprint(ne_counts(dev))
    print("Test:")
    print(ts[2], ns[2])
    pprint(ne_counts(test))

    write_data(train, f"{out_folder}/train.jsonl")
    write_data(dev, f"{out_folder}/dev.jsonl")
    write_data(test, f"{out_folder}/test.jsonl")

    split_ids: Dict[str, List[str]] = {}
    split_ids["train"] = [sen["id"] for sen in train]
    split_ids["dev"] = [sen["id"] for sen in dev]
    split_ids["test"] = [sen["id"] for sen in test]
    with open(f"{out_folder}/split_ids.json", "w") as f:
        json.dump(split_ids, f)

    return None


def split_by_ids(sentences: Iterable[Sentence],
                 split_ids: Dict[str, List[str]],
                 out_folder: str) -> None:
    train: List[Sentence] = []
    dev: List[Sentence] = []
    test: List[Sentence] = []
    for sentence in sentences:
        if sentence["id"] in split_ids["dev"]:
            dev.append(sentence)
        elif sentence["id"] in split_ids["test"]:
            test.append(sentence)
        elif sentence["id"] in split_ids["train"]:
            train.append(sentence)
        # else:
        #     print("This should not happen")
        #     print(sentence)
    write_data(train, f"{out_folder}/train.jsonl")
    write_data(dev, f"{out_folder}/dev.jsonl")
    write_data(test, f"{out_folder}/test.jsonl")

    return None


def filter_trash(sentences: Iterable[Sentence],
                 min_chars: int = 10
                 ) -> Iterable[Sentence]:
    seen: Set[str] = set()
    for sentence in sentences:
        raw = " ".join(sentence["tokens"])
        if raw in seen:
            pass
        elif len(raw) < min_chars \
                and all(tag == "O" for tag in sentence["ner_tags"]):
            seen.add(raw)
        else:
            seen.add(raw)
            yield sentence


def get_sizes(n: int) -> Tuple[int, int, int]:
    # train: 0.64 dev: 0.16 test: 0.2
    test_size = int(n * 0.2)
    dev_size = int(n * 0.16)
    train_size = n - test_size - dev_size
    return train_size, dev_size, test_size


def ne_counts(sentences: Iterable[Sentence]
              ) -> Tuple[int, int, Dict[str, int]]:
    total = 0
    has_ne = 0
    ne_dict: Dict[str, int] = {}
    for sentence in sentences:
        total += 1
        if any(tag != "O" for tag in sentence["ner_tags"]):
            has_ne += 1
            for tag in filter(lambda x: x != "O", sentence["ner_tags"]):
                if tag not in ne_dict:
                    ne_dict[tag] = 0
                ne_dict[tag] += 1
    return total, has_ne, ne_dict


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--outfolder")
    parser.add_argument("--split_ids")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--count_ne", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    random.seed(args.seed)
    if args.count_ne:
        t, n, d = ne_counts(filter_trash(read_json(args.infile)))
        print(f"{t}, {n}, {n/t:.2f}")
        pprint(sorted(d.items(), key=lambda x: x[1]))
        pprint([(k, get_sizes(v)) for k, v in sorted(d.items(), key=lambda x: x[1]) if k.startswith("B")])
        total_sizes = get_sizes(t)
        ne_sizes = get_sizes(n)
        print(total_sizes, ne_sizes)
    elif args.split_ids:
        split_by_ids(read_json(args.infile),
                     json.load(open(args.split_ids)),
                     args.outfolder)
    elif args.outfolder:
        split_sort(filter_trash(read_json(args.infile)), args.outfolder)


if __name__ == "__main__":
    main()
