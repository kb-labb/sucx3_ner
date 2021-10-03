#!/usr/bin/env python
# coding=utf-8

"""
a little script transforming conllu to json
"""

from typing import List, Iterable, Any, Dict, NamedTuple
import json
from colorama import Fore, Back, Style


fore = [Fore.BLACK, Fore.BLUE, Fore.CYAN, Fore.GREEN, Fore.LIGHTBLACK_EX,
        Fore.LIGHTBLUE_EX, Fore.LIGHTCYAN_EX, Fore.LIGHTGREEN_EX,
        Fore.LIGHTMAGENTA_EX, Fore.LIGHTRED_EX, Fore.LIGHTWHITE_EX,
        Fore.LIGHTYELLOW_EX, Fore.MAGENTA, Fore.RED, Fore.WHITE, Fore.YELLOW]
back = [Back.BLACK, Back.BLUE, Back.CYAN, Back.GREEN, Back.LIGHTBLACK_EX,
        Back.LIGHTBLUE_EX, Back.LIGHTCYAN_EX, Back.LIGHTGREEN_EX,
        Back.LIGHTMAGENTA_EX, Back.LIGHTRED_EX, Back.LIGHTWHITE_EX,
        Back.LIGHTYELLOW_EX, Back.MAGENTA, Back.RED, Back.WHITE, Back.YELLOW]
# print(" ".join([c + str(i) + Style.RESET_ALL for i, c in enumerate((fore))]))
# print(" ".join([c + Fore.WHITE + str(i) + Style.RESET_ALL for i, c in enumerate((back))]))

Sentence = NamedTuple("Sentence", [("id", str), ("tokens", List[List[str]])])


def read_conllu(fn: str) -> Iterable[Sentence]:
    with open(fn) as fh:
        tokens: List[List[str]] = []
        for line in fh:
            if line.startswith("#"):
                if line.startswith("# sent_id"):
                    _id = line.split("=")[1].strip()
                pass
            elif line == "\n":
                yield Sentence(_id, tokens)
                tokens = []
            else:
                tokens.append(line.split())


def debug_print(tokens: List[List[str]]) -> None:
    for token in tokens:
        if len(token) == 10:
            print("\t".join(token))
        elif len(token) > 10:
            print(fore[13], "\t".join(token), Style.RESET_ALL)
        elif len(token) < 10:
            print(fore[1], "\t".join(token), Style.RESET_ALL)


def conllu_to_json(fn_conllu: str, fn_json: str) -> None:
    with open(fn_json, "w") as fh:
        for i, sentence in enumerate(read_conllu(fn_conllu)):
            jsen: Dict[str, Any] = {"id": sentence.id}
            if not all(len(x) == 10 for x in sentence.tokens):
                # debug_print(sentence.tokens)
                # continue
                ns = sentence.tokens[:]
                for i in range(len(ns)):
                    if len(ns[i]) < 10:
                        t = ns[i][0:2] + ["/"] + ns[i][2:]
                        ns[i] = t
                sentence = Sentence(sentence.id, ns)
                assert all(len(x) == 10 for x in sentence.tokens)
            columns = list(zip(*sentence.tokens))
            tokens = list(columns[1])
            pos = list(columns[3])
            ner = ["-".join(x) if x[0] != "O" else "O" for x in
                   # zip(columns[10], columns[11])]  # sic
                   zip(columns[8], columns[9])]  # suc
            jsen["tokens"] = tokens
            jsen["pos_tags"] = pos
            jsen["ner_tags"] = ner
            print(json.dumps(jsen), file=fh)


if __name__ == "__main__":
    import sys
    conllu_to_json(sys.argv[1], sys.argv[2])
