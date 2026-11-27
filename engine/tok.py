"""Text <-> token ids for tuff_engine run (the engine takes and prints ids; tokenizing is not the hot loop).

    python engine/tok.py encode <snapshot> "The history of the transistor"     -> ids
    python engine/tok.py decode <snapshot> "785 3840 315 ..."                   -> text
    tuff_engine run --model <snapshot> --store experts/ --ids "$(python engine/tok.py encode <snapshot> "...")"
"""
import sys

from transformers import AutoTokenizer

if __name__ == "__main__":
    cmd, snap, arg = sys.argv[1:4]
    tok = AutoTokenizer.from_pretrained(snap)
    if cmd == "encode":
        print(" ".join(map(str, tok(arg)["input_ids"])))
    else:
        print(tok.decode([int(x) for x in arg.split()]))
