# build_alpaca_singleturn.py
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def make_user_text(instruction: str, input_: str) -> str:
    instruction = (instruction or "").strip()
    input_ = (input_ or "").strip()
    if input_:
        return f"{instruction}\n\n{input_}"
    return instruction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/alpaca_singleturn", help="output directory")
    ap.add_argument("--n", type=int, default=3000, help="number of samples to export")
    ap.add_argument("--split", type=str, default="train", help="dataset split")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = load_dataset("tatsu-lab/alpaca", split=args.split)

    n = min(args.n, len(ds))

    dict_path = out_dir / "alpaca_singleturn_dict.jsonl"
    list_path = out_dir / "alpaca_singleturn_list.jsonl"

    cnt = 0
    with dict_path.open("w", encoding="utf-8") as f_dict, list_path.open("w", encoding="utf-8") as f_list:
        for ex in ds.select(range(n)):
            instruction = ex.get("instruction", "")
            input_ = ex.get("input", "")
            output = ex.get("output", "")

            user_text = make_user_text(instruction, input_)
            assistant_text = (output or "").strip()

            if not user_text or not assistant_text:
                continue

            # single-turn dict version
            dict_item = {
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            }

            # single-turn list version
            list_item = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]

            f_dict.write(json.dumps(dict_item, ensure_ascii=False) + "\n")
            f_list.write(json.dumps(list_item, ensure_ascii=False) + "\n")
            cnt += 1

    print(f"Exported {cnt} samples")
    print(f" - dict jsonl: {dict_path}")
    print(f" - list jsonl: {list_path}")


if __name__ == "__main__":
    main()
