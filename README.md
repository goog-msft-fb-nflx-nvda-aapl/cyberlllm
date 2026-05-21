# CyberLLM: Multi-Stage Fine-Tuning for Cybersecurity Threat Intelligence

**Course:** LLM Applications in Cybersecurity  
**Student:** James Watanabe (R13921031)

## Project Overview

Domain-adapted Qwen2.5-7B-Instruct for NIST Cybersecurity Framework classification via LoRA SFT + DPO alignment on 8× NVIDIA H200 GPUs.

**Best result:** SFT rank=64 achieves **83% accuracy** on MMLU Computer Security (+9 pp over baseline).

## Results

| Model | MMLU CS Accuracy |
|-------|:---:|
| Baseline (Qwen2.5-7B-Instruct) | 74.0% |
| SFT rank=8 | 74.0% |
| SFT rank=16 | 76.0% |
| SFT rank=32 | 75.0% |
| **SFT rank=64** | **83.0%** |
| DPO rank=32 | 75.0% |
| DPO rank=64 | 81.0% |

## Repository Structure

```
configs/          LlamaFactory YAML configs (SFT + DPO, all ranks)
data/             Dataset preparation pipeline
eval/             MMLU evaluation script
scripts/          Training launch scripts (8-GPU torchrun)
logs/             Training metric summaries
reports/          Progress reports (Week 14, 15, 16) + loss curves
reports/eval_results/  Per-model evaluation JSON files
```

## Training (on gsm-gpu server)

```bash
# 1. Prepare data
python data/prepare_data.py

# 2. SFT training (8 GPUs, rank=64)
export PATH=/home/jtan/miniconda3/envs/rl_hw3/bin:$PATH
cd /home/jtan/llm/LLaMA-Factory
FORCE_TORCHRUN=1 NPROC_PER_NODE=8 llamafactory-cli train configs/cyberlllm_sft_rank64.yaml

# 3. DPO alignment
FORCE_TORCHRUN=1 NPROC_PER_NODE=8 llamafactory-cli train configs/cyberlllm_dpo_rank64.yaml

# 4. Evaluate
python eval/evaluate.py --model_path /path/to/Qwen2.5-7B-Instruct \
    --adapter_path /path/to/saves/cyberlllm_sft_rank64 \
    --output results.json
```

## Hardware

- 8× NVIDIA H200 (143GB VRAM each, ~1.1TB total)
- Training time: ~5 min per SFT run, ~2 min per DPO run
- Framework: LlamaFactory v0.9.5, PyTorch 2.4.0+cu124
