# IdeaReward: Benchmarking and Training Reward Models for Research Idea Generation

<p align="center">
    <a href=""><img alt="Paper" src="https://img.shields.io/badge/📖-Paper-orange"></a>
    <a href="https://huggingface.co/collections/THU-KEG/ideareward"><img alt="Hugging Face" src="https://img.shields.io/badge/🤗HuggingFace-IdeaReward-green"></a>
    <a href="https://github.com/THU-KEG"><img alt="THU-KEG Group" src="https://img.shields.io/badge/THU--KEG_Group-blueviolet"></a>
</p>

<p align="justify">

Research idea generation aims to produce research ideas that can be developed into publication-worthy research projects. However, as large language model (LLM)-based systems generate research ideas at scale, the central bottleneck shifts from producing candidate ideas to reliably evaluating, selecting, and improving them before costly research execution. Reward models offer a promising solution for this goal by learning to assign quality-aware reward scores from preference data. In this paper, we present a systematic study of reward modeling for research idea generation. We present IdeaRewardBench, the first multi-signal, living benchmark for research idea evaluation. IdeaRewardBench contains 3,056 idea preference instances grounded in complementary signals of idea quality, and is updated semiannually to reflect the latest research frontier. We evaluate a comprehensive set of 23 baselines on the full IdeaRewardBench, and find that all baselines struggle to distinguish high-quality research ideas: the best baseline achieves only 62.8% pairwise accuracy on the review-grounded split and 60.6% on the impact-grounded split, where 50% is the random baseline. We then present IdeaReward, the first reward model for research idea generation, trained on 19,670 high-quality idea preference pairs constructed by a fully automated pipeline. IdeaReward establishes state-of-the-art performance on IdeaRewardBench, outperforming baselines based on much larger LLMs. We further show that IdeaReward improves downstream test-time best-of-N idea selection. We will release our data, models, and code upon publication.

</p>

## Evaluation

Evaluate IdeaReward-Llama and IdeaReward-Qwen on IdeaRewardBench. `BENCHMARK_DIR` should contain the downloaded benchmark folders.

```bash
pip install -e train/OpenRLHF
python evaluation/evaluate_ideareward.py \
  --benchmark BENCHMARK_DIR \
  --llama-checkpoint LLAMA_CHECKPOINT \
  --qwen-checkpoint QWEN_CHECKPOINT \
  --output-dir results
```

## Training

Train the Llama-3.1-8B-Instruct and Qwen3-8B reward models on the downloaded preference JSONL file.

```bash
pip install -e train/OpenRLHF
bash train/train_llama.sh TRAIN.jsonl LLAMA_MODEL LLAMA_OUTPUT
bash train/train_qwen.sh TRAIN.jsonl QWEN_MODEL QWEN_OUTPUT
```
