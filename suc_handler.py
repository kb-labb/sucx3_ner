#!/usr/bin/env python
# coding: utf-8

import xml
import xml.etree.ElementTree as ET
import argparse
import json
from conllu2json import conllu_to_json
from typing import List, Optional, Tuple

# take only sentences where the mapping fits perfectly
# ignore other because it is a bit mixed
# use an additional file that lets countries, regions, cities that are
# inst to be mapped to LOC
tag_map = {"person": "PRS",
           "place": "LOC",
           "inst": "ORG",
           "work": "WRK",
           "product": "OBJ",
           # "other": "MISC",
           "animal": "PRS",
           "event": "EVN",
           "myth": "PRS",
           }

with open("inst_LOC.json") as f:
    inst_LOC = json.load(f)


def get_tags(sentence, tokens):
    pos, names, tags = zip(*[(i, token, name.attrib["type"]) for name in sentence.findall(".//name")
                             for i, token in enumerate(name.findall(".//w"))])
    bios, _tags = [], []
    for i, token in enumerate(tokens):
        try:
            j = names.index(token)
            bio = "B" if pos[j] == 0 else "I"
            tag = tags[j]
        except ValueError:
            tag = "_"
            bio = "O"
        bios.append(bio)
        _tags.append(tag)
        # TODO check that combo is allowed
    assert len(tokens) == len(_tags)
    return bios, _tags


def check_tag_match(sentence: xml.etree.ElementTree.Element,
                    tokens: List[xml.etree.ElementTree.Element],
                    no_MSR_TME: bool = False
                    ) -> Tuple[List[Optional[str]], List[Optional[str]]]:
    pos_a, names_a, tags_a = zip(*[(i, token, name.attrib["type"]) for name in sentence.findall(".//name")
                                   for i, token in enumerate(name.findall(".//w"))])
    pos_b, names_b, tags_b = zip(*[(i, token, name.attrib["type"]) for name in sentence.findall(".//ne")
                                   for i, token in enumerate(name.findall(".//w"))])
    bio: List[Optional[str]] = []
    tags: List[Optional[str]] = []
    for token in tokens:
        try:
            j_a = names_a.index(token)
            bio_a = "B" if pos_a[j_a] == 0 else "I"
            tag_a = tags_a[j_a]
        except ValueError:
            tag_a = None
            bio_a = "O"
        try:
            j_b = names_b.index(token)
            bio_b = "B" if pos_b[j_b] == 0 else "I"
            tag_b = tags_b[j_b]
        except ValueError:
            tag_b = None
            bio_b = "O"

        # check that combo is allowed
        if tag_a is None and (tag_b == "MSR" or tag_b == "TME"):
            if no_MSR_TME:
                tags.append("_")
                bio.append("O")
            else:
                tags.append(tag_b)
                bio.append(bio_b)
        elif tag_a in tag_map and tag_map[tag_a] == tag_b and bio_a == bio_b:
            tags.append(tag_b)
            bio.append(bio_a)
        elif tag_a == "inst" and \
                tag_b == "LOC" and \
                inst_LOC.get(token.text, None) == ["inst", "LOC"] and \
                bio_a == bio_b:
            tags.append("LOC")
            bio.append(bio_a)
        elif tag_a in tag_map and tag_b is None:
            tags.append(tag_map[tag_a])
            bio.append(bio_a)
        elif tag_a is None and tag_b is None:
            # print(token.text, tag_a, tag_b)
            tags.append("_")
            bio.append("O")
        else:
            # print(token.text, tag_a, bio_a, tag_b, bio_b)
            tags.append(None)
            bio.append(None)

    assert len(tokens) == len(tags)
    return bio, tags


def convert_suc_to_conllu(fn_in: str,
                          fn_out: str,
                          original_tags: bool,
                          no_MSR_TME: bool) -> None:

    tree = ET.parse(fn_in)
    root = tree.getroot()

    sen_with_ne = 0
    failed = 0
    total = 0

    with open(fn_out, "w") as fh:
        for text in root:
            _id = text.attrib["id"]
            print(f"# newdoc id = {_id}", file=fh)
            for sentence in text:
                total += 1
                tokens = [token for token in sentence.findall(".//w")]
                try:
                    if original_tags:
                        bios, tags = get_tags(sentence, tokens)
                        sen_with_ne += 1
                    else:
                        bios, tags = check_tag_match(sentence, tokens, no_MSR_TME)
                        sen_with_ne += 1
                        if None in tags:
                            failed += 1
                            continue
                except ValueError:
                    bios, tags = zip(*[("O", "_") for i in range(len(tokens))])
                _text = " ".join([t.text for t in tokens])
                _id = sentence.attrib["id"]
                print(f"# sent_id = {_id}", file=fh)
                print(f"# text = {_text}", file=fh)
                for i, token in enumerate(tokens):
                    bio = bios[i]
                    ne_tag = tags[i]
                    print("\t".join([str(i + 1),
                                     token.text,
                                     token.attrib["lemma"].strip("|"),
                                     token.attrib["pos"],
                                     token.attrib["msd"].replace(".", "|"),
                                     "_",
                                     str(int(token.attrib.get("dephead", "0"))),
                                     token.attrib["deprel"], bio, ne_tag
                                     ]),
                          file=fh)

                print("", file=fh)
    print(sen_with_ne, failed, total)
    return None


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--outfile")
    parser.add_argument("--original_tags", action="store_true")
    parser.add_argument("--no_MSR_TME", action="store_true")
    parser.add_argument("--conllu2json", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    if args.conllu2json:
        conllu_to_json(args.infile, args.outfile)
    else:
        convert_suc_to_conllu(args.infile,
                              args.outfile,
                              args.original_tags,
                              args.no_MSR_TME)
    return None


if __name__ == "__main__":
    main()
