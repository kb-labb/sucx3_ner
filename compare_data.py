import json
import argparse
import random
from typing import Iterable, List, Union, Tuple, TypeVar, Any, Dict
from typing_extensions import TypedDict
from tqdm import tqdm
from pprint import pprint


A = Union[int, List[str]]
B = Tuple[str, int, List[str], List[str]]
C = TypeVar("C")
Sentence = TypedDict("Sentence", {"id": int,
                                  "tokens": List[str],
                                  "ner_tags": List[str],
                                  "pos_tags": List[str]
                                  }
                     )


def del_list_indexes(input_list: List[C], id_to_del: List[int]) -> List[C]:
    _id_to_del = set(id_to_del)
    somelist = [i for j, i in enumerate(input_list) if j not in _id_to_del]
    return somelist


def read_json(fn: str) -> Iterable[Sentence]:
    with open(fn) as fh:
        for line in fh:
            yield json.loads(line)


def collect_tuples(fns: List[str]) -> List[B]:
    data = []
    for fn in fns:
        for sentence in read_json(fn):
            data.append((fn, sentence["id"], sentence["tokens"], sentence["ner_tags"]))
    return data


def compare_data(data_a: List[B],
                 data_b: List[B]
                 ) -> Tuple[List[B], List[B], List[Tuple[B, B]]]:
    both: List[Tuple[B, B]] = []
    to_pop_outer = []
    for i, a in tqdm(enumerate(data_a), total=len(data_a)):
        to_pop_inner: List[int] = []
        for j, b in enumerate(data_b):
            if a[2] == b[2]:
                to_pop_outer.append(i)
                to_pop_inner.append(j)
                both.append((a, b))
        data_b = del_list_indexes(data_b, to_pop_inner)
    data_a = del_list_indexes(data_a, to_pop_outer)
    return data_a, data_b, both


def remove_doubles(fns: List[str]) -> List[Sentence]:
    all_data = [x for fn in fns for x in read_json(fn)]
    sentence_dict = {tuple(sentence["tokens"]): sentence
                     for sentence in all_data}
    return list(sentence_dict.values())


def remove_doubles_single_file(fn: str) -> List[Sentence]:
    sentence_dict = {tuple(sentence["tokens"]): sentence
                     for sentence in read_json(fn)}
    return list(sentence_dict.values())


def split_data(data: List[Sentence]
               ) -> Tuple[List[Sentence], List[Sentence], List[Sentence]]:
    total = len(data)
    test_len = int(total * 0.2)
    train_len = int((total - test_len) * 0.8)
    dev_len = total - test_len - train_len
    random.shuffle(data)
    test = data[0:test_len]
    dev = data[test_len:test_len + dev_len]
    train = data[test_len + dev_len:]
    return train, dev, test


def write_data(data: List[Any], fn: str) -> None:
    with open(fn, "w") as fh:
        for element in data:
            print(json.dumps(element), file=fh)


def map_tags(both_fn: str) -> Tuple[Dict[str, Dict[str, int]]]:
    a2b = {}
    b2a = {}
    with open(both_fn) as fh:
        for line in fh:
            jline = json.loads(line)
            tags_a = jline[0][3]
            tags_b = jline[1][3]
            for a, b in zip(tags_a, tags_b):
                if a not in a2b:
                    a2b[a] = {}
                if b not in a2b[a]:
                    a2b[a][b] = 0
                a2b[a][b] += 1
                if b not in b2a:
                    b2a[b] = {}
                if a not in b2a[b]:
                    b2a[b][a] = 0
                b2a[b][a] += 1
    return a2b, b2a



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files_a", nargs="+")
    parser.add_argument("--files_b", nargs="+")
    parser.add_argument("--remove_doubles", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--map_tags", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    assert sum([args.remove_doubles, args.compare, args.map_tags]) == 1

    if args.remove_doubles:
        no_doubles = remove_doubles(args.files_a)
        train, dev, test = split_data(no_doubles)
        for d, fn in zip([train, dev, test], args.files_b):
            write_data(d, fn)
    elif args.compare:
        data_a = collect_tuples(args.files_a)
        data_b = collect_tuples(args.files_b)
        rest_a, rest_b, both = compare_data(data_a, data_b)

        write_data(rest_a, "rest_a.jsonl")
        write_data(rest_b, "rest_b.jsonl")
        write_data(both, "both.jsonl")

        print(f"rest_a: {len(rest_a)}\nrest_b: {len(rest_b)}\nboth: {len(both)}")
    elif args.map_tags:
        both = args.files_a[0]
        a2b, b2a = map_tags(both)
        pprint(a2b)
        pprint(b2a)




if __name__ == "__main__":
    main()
