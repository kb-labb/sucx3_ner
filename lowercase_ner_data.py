import json
import argparse
from typing import Iterable, Dict, List, Union

A = Union[int, List[str]]


def lowercase_all_text(fn: str, both: bool) -> Iterable[Dict[str, A]]:
    with open(fn) as fh:
        i = 0
        for line in fh:
            jline = json.loads(line)
            if both:
                jline["id"] = i
                yield jline
                i += 1
            jline["tokens"] = [token.lower() for token in jline["tokens"]]
            jline["id"] = i
            yield jline
            i += 1


def lowercase_ne_only(fn: str, both: bool) -> Iterable[Dict[str, A]]:
    with open(fn) as fh:
        i = 0
        for line in fh:
            jline = json.loads(line)
            if both:
                jline["id"] = i
                yield jline
                i += 1
            tokens = []
            for tag, token in zip(jline["ner_tags"], jline["tokens"]):
                if tag != "O":
                    token = token.lower()
                tokens.append(token)
            jline["tokens"] = tokens
            jline["id"] = i
            yield jline
            i += 1


def lowercase_wrapper(fn_in: str, ne_only: bool, both: bool, fn_out) -> None:
    with open(fn_out, "w") as fout:
        if ne_only:
            for jline in lowercase_ne_only(fn_in, both):
                print(json.dumps(jline), file=fout)
        else:
            for jline in lowercase_all_text(fn_in, both):
                print(json.dumps(jline), file=fout)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infiles", nargs="+")
    parser.add_argument("--outfiles", nargs="+")
    parser.add_argument("--ne_only", action="store_true")
    parser.add_argument("--both", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    print(args)
    assert len(args.infiles) == len(args.outfiles)
    for i, o in zip(args.infiles, args.outfiles):
        lowercase_wrapper(i, args.ne_only, args.both, o)


if __name__ == "__main__":
    main()
