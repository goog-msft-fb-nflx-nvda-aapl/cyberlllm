"""
Data preparation for CyberLLM project.
Combines cybersec datasets and MMLU computer_security into LlamaFactory format.
"""
import json, random, os
from pathlib import Path

random.seed(42)
LLAMA_DATA = Path("/home/jtan/llm/LLaMA-Factory/data")
NIST_LABELS = ["identify", "protect", "detect", "respond", "recover", "not applicable", "none"]

def load_cybersec_alpaca():
    data = json.loads((LLAMA_DATA / "cybersec_alpaca.json").read_text())
    print(f"cybersec_alpaca: {len(data)} samples")
    return data

def harmony_to_alpaca(conv_sample):
    convs = conv_sample["conversations"]
    human_msg = next((c["value"] for c in convs if c["from"] == "human"), "")
    gpt_msg = next((c["value"] for c in convs if c["from"] == "gpt"), "")
    sys_clean = "You are an expert cybersecurity analyst. Analyze the text using the NIST Cybersecurity Framework. Reason step by step."
    return {
        "instruction": f"{sys_clean}\n\n{human_msg[:2000].strip()}",
        "input": "",
        "output": gpt_msg[:3000].strip()
    }

def load_cybersec_harmony():
    raw = json.loads((LLAMA_DATA / "cybersec_harmony" / "train.json").read_text())
    data = [harmony_to_alpaca(s) for s in raw]
    print(f"cybersec_harmony train: {len(data)} samples")
    return data

def load_mmlu_cs():
    from datasets import load_dataset
    letters = {0:"A", 1:"B", 2:"C", 3:"D"}
    results = []
    for split in ["test", "validation", "dev"]:
        try:
            ds = load_dataset("cais/mmlu", "computer_security", split=split)
            for ex in ds:
                c = ex["choices"]; a = ex["answer"]
                opts = "\n".join(f"{letters[i]}. {v}" for i,v in enumerate(c))
                results.append({
                    "instruction": f"Answer this cybersecurity question.\n\nQuestion: {ex["question"]}\n\nOptions:\n{opts}",
                    "input": "",
                    "output": f"The correct answer is {letters[a]}: {c[a]}",
                    "_split": split, "_answer": letters[a]
                })
        except Exception as e:
            print(f"  MMLU {split}: {e}")
    print(f"MMLU computer_security: {len(results)} samples")
    return results

def load_mmlu_cs_test_raw():
    from datasets import load_dataset
    letters = {0:"A", 1:"B", 2:"C", 3:"D"}
    ds = load_dataset("cais/mmlu", "computer_security", split="test")
    results = []
    for ex in ds:
        c = ex["choices"]; a = ex["answer"]
        results.append({"question": ex["question"], "choices": c, "answer": a, "answer_letter": letters[a]})
    return results

def generate_dpo_pairs(alpaca_data):
    pairs = []
    for s in alpaca_data:
        out = s["output"]
        if "label is" not in out.lower():
            continue
        true_label = next((l for l in NIST_LABELS if f"label is {l}" in out.lower()), None)
        if not true_label:
            continue
        wrong = random.choice([l for l in NIST_LABELS if l != true_label])
        rejected = (
            f"The label is {wrong.title()}. This sentence broadly relates to {wrong.title()} "
            f"activities as it involves concepts commonly associated with this NIST CSF category."
        )
        pairs.append({"instruction": s["instruction"], "input": s.get("input",""), "chosen": out, "rejected": rejected})
    print(f"DPO pairs: {len(pairs)}")
    return pairs

def clean(d):
    return {"instruction": d["instruction"], "input": d.get("input",""), "output": d["output"]}

def main():
    alpaca = load_cybersec_alpaca()
    harmony = load_cybersec_harmony()
    mmlu = load_mmlu_cs()

    all_train = [clean(d) for d in alpaca + harmony + mmlu]
    random.shuffle(all_train)
    val_n = max(100, int(len(all_train)*0.05))
    val_data, train_data = all_train[:val_n], all_train[val_n:]

    print(f"\nSplits -> train:{len(train_data)}  val:{len(val_data)}")
    (LLAMA_DATA / "cyberlllm_train.json").write_text(json.dumps(train_data, indent=2, ensure_ascii=False))
    print("Saved cyberlllm_train.json")

    mmlu_raw = load_mmlu_cs_test_raw()
    (LLAMA_DATA / "mmlu_cs_eval.json").write_text(json.dumps(mmlu_raw, indent=2, ensure_ascii=False))
    print(f"Saved mmlu_cs_eval.json ({len(mmlu_raw)} test samples)")

    dpo_pairs = generate_dpo_pairs(alpaca)
    (LLAMA_DATA / "cyberlllm_dpo.json").write_text(json.dumps(dpo_pairs, indent=2, ensure_ascii=False))
    print("Saved cyberlllm_dpo.json")

    print("\nAll done!")

if __name__ == "__main__":
    main()
