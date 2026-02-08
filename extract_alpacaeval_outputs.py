# tools/extract_router_outputs.py

# This script extracts the final outputs from the router logs and saves them in a format compatible with AlpacaEval.
# It reads the router log (jsonl), processes each record, and writes a new json
import json, argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router_log", type=str, required=True)  # your router output jsonl
    ap.add_argument("--out", type=str, required=True)         # final_outputs jsonl
    ap.add_argument("--generator_name", type=str, default="router")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n = 0
    with open(args.router_log, "r", encoding="utf-8") as rf, open(args.out, "w", encoding="utf-8") as wf:
        for line in rf:
            if not line.strip():
                continue
            obj = json.loads(line)
            rec = {
                "id": int(obj["t"]),
                "generator": args.generator_name,
                "prompt": obj["prompt"],
                "output": obj["final"],
                "meta": {
                    "chosen_names": obj.get("chosen_names", []),
                    "cost": obj.get("cost", None),
                },
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"[DONE] wrote {n} lines to {args.out}")

if __name__ == "__main__":
    main()