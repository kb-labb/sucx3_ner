import json
import argparse
import random
from typing import Iterable, TypedDict, List

Sentence = TypedDict("Sentence",
                     {
                        "id": str,
                        "tokens": List[str],
                        "pos_tags": List[str],
                        "ner_tags": List[str],
                     }
                     )


def lowercase_all_text(fn: str, both: bool, mix: bool) -> Iterable[Sentence]:
    with open(fn) as fh:
        for line in fh:
            jline = json.loads(line)
            lower_jline = {k: v for k, v in jline.items()}  # type: Sentence
            lower_jline["tokens"] = [token.lower() for token in jline["tokens"]]
            lower_jline["id"] = jline["id"] + "_lower"
            if both:
                yield jline
                yield lower_jline
            elif mix:
                if random.random() <= 0.5:
                    yield jline
                else:
                    yield lower_jline
            else:
                yield lower_jline


def lowercase_ne_only(fn: str, both: bool, mix: bool) -> Iterable[Sentence]:
    with open(fn) as fh:
        for line in fh:
            jline = json.loads(line)
            tokens = []
            for tag, token in zip(jline["ner_tags"], jline["tokens"]):
                if tag != "O":
                    token = token.lower()
                tokens.append(token)
            lower_jline = {k: v for k, v in jline.items()}  # type: Sentence
            lower_jline["tokens"] = tokens
            lower_jline["id"] = jline["id"] + "_ne_lower"
            if both:
                yield jline
                yield lower_jline
            elif mix:
                if random.random() <= 0.5:
                    yield jline
                else:
                    yield lower_jline
            else:
                yield lower_jline


def lowercase_wrapper(fn_in: str,
                      ne_only: bool,
                      both: bool,
                      mix: bool,
                      fn_out,
                      ) -> None:
    with open(fn_out, "w") as fout:
        if ne_only:
            for jline in lowercase_ne_only(fn_in, both, mix):
                print(json.dumps(jline), file=fout)
        else:
            for jline in lowercase_all_text(fn_in, both, mix):
                print(json.dumps(jline), file=fout)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infiles", nargs="+")
    parser.add_argument("--outfiles", nargs="+")
    parser.add_argument("--ne_only", action="store_true")
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--mix", action="store_true")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    if args.seed:
        random.seed(args.seed)
    print(args)
    assert len(args.infiles) == len(args.outfiles)
    for i, o in zip(args.infiles, args.outfiles):
        lowercase_wrapper(i, args.ne_only, args.both, args.mix, o)


if __name__ == "__main__":
    main()
