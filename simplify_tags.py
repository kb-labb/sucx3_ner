import json
import argparse
from typing import Callable, List, Iterable, TextIO, TypedDict

Sentence = TypedDict("Sentence", {"id": str,
                                  "tokens": List[str],
                                  "pos_tags": List[str],
                                  "ner_tags": List[str]})


def read_tag_file(fn: str) -> Iterable[Sentence]:
    with open(fn) as fh:
        for line in fh:
            yield json.loads(line)


def modify_sentence(sentence: Sentence,
                    key: str,
                    modification: Callable
                    ) -> Sentence:
    sentence[key] = modification(sentence[key])
    return sentence


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--outfile")
    return parser.parse_args()


def main() -> None:
    args = get_args()

    work_key = "ner_tags"
    def remove_BI(tags: List[str]) -> List[str]:
        return [t[2:] if len(t) > 1 else t for t in tags]

    with open(args.outfile, "w") as fout:
        for sentence in read_tag_file(args.infile):
            new_sentence = modify_sentence(sentence, work_key, remove_BI)
            print(json.dumps(new_sentence), file=fout)
    

if __name__ == "__main__":
    main()