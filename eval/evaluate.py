"""
Evaluation script for CyberLLM models.
Evaluates on MMLU computer_security (0-shot accuracy).

Usage:
  python evaluate.py --model_path /path/to/model [--adapter_path /path/to/lora]
"""
import json, argparse, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

EVAL_FILE = Path("/home/jtan/llm/LLaMA-Factory/data/mmlu_cs_eval.json")
LETTERS = ["A", "B", "C", "D"]

def format_prompt(sample, tokenizer):
    q = sample["question"]
    choices = sample["choices"]
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    user_msg = f"Answer this cybersecurity question by selecting A, B, C, or D.\n\nQuestion: {q}\n\nOptions:\n{opts}\n\nAnswer with just the letter."
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def load_model(model_path, adapter_path=None):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return tokenizer, model

def predict_letter(text):
    for letter in LETTERS:
        if text.strip().startswith(letter):
            return letter
    # fallback: find first occurrence
    for letter in LETTERS:
        if letter in text:
            return letter
    return "A"  # default

def evaluate(tokenizer, model, eval_data, max_new_tokens=10):
    correct = 0
    results = []
    for i, sample in enumerate(eval_data):
        prompt = format_prompt(sample, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0, pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = predict_letter(generated)
        gold = sample["answer_letter"]
        is_correct = (pred == gold)
        correct += int(is_correct)
        results.append({"idx": i, "pred": pred, "gold": gold, "correct": is_correct, "generated": generated.strip()})
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(eval_data)}] Running accuracy: {correct/(i+1)*100:.1f}%")

    accuracy = correct / len(eval_data)
    print(f"\nFinal accuracy: {correct}/{len(eval_data)} = {accuracy*100:.2f}%")
    return accuracy, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to base model")
    parser.add_argument("--adapter_path", default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--output", default="eval_results.json", help="Save results JSON")
    args = parser.parse_args()

    eval_data = json.loads(EVAL_FILE.read_text())
    print(f"Loaded {len(eval_data)} evaluation samples")

    tokenizer, model = load_model(args.model_path, args.adapter_path)
    accuracy, results = evaluate(tokenizer, model, eval_data)

    out = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "num_samples": len(eval_data),
        "accuracy": accuracy,
        "correct": sum(r["correct"] for r in results),
        "results": results
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
