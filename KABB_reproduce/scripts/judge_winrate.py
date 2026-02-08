import json
import pandas as pd
from alpaca_eval import evaluate
import os
from datasets import load_dataset


os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_KEY"] = "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"

print("正在加载基准数据...")
try:
    
    ref_ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval_gpt4_baseline")
    reference_outputs = pd.DataFrame(ref_ds["eval"])
except Exception as e:
    print(f"无法在线加载，尝试使用基础配置: {e}")
    ref_ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")
    reference_outputs = pd.DataFrame(ref_ds["eval"])



with open("outputs/kabb_alpacaeval_outputs.json", "r", encoding="utf-8") as f:
    raw = json.load(f)   # ✅ 读整个 list

    model_outputs = []
    for data in raw:
        model_outputs.append({
            "instruction": data["instruction"],   # 注意字段名
            "output": data["output"],
            "generator": data.get("generator", "your_router")
        })


result = evaluate(
    model_outputs=model_outputs,
    reference_outputs=reference_outputs,
    annotators_config="alpaca_eval_gpt4_turbo_fn"
)

