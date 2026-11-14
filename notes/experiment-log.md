# Historical Experiment Log: Bilingual Training

> 历史保留：本文件记录旧 TinyStories 英文 + 中文 Wikipedia 混合方案。
> 当前正式英文单语训练的原始日志在 `experiments/english_single/base/tensorboard/train.jsonl`，
> 结论见 `technical-report.md`。下列旧指标不与新英文指标合并。

Record the data version, tokenizer version, configuration, token count, validation loss, perplexity, throughput, and fixed-prompt samples for each training run.

## Pilot run

- Configuration: 8 layers, d_model 512, 8 heads, context 512, 16k SentencePiece vocabulary.
- Training data: 112,968,440 train tokens; validation data: 2,401,669 tokens.
- Effective batch: 8,192 tokens per optimizer step.
- Pilot target: 5 million seen tokens.
- Initial train loss: 9.795 at step 1.
- Validation loss at step 250: 4.780; at step 500: 3.941. These early values used the English-first validation method and must not be compared with later weighted metrics.
- Output after pilot: coherent TinyStories-like sentence structure for the fixed English prompt, with repetition and logical errors.
- Resume check: step 611 checkpoint resumed and reached step 623.

## Long run and measurement changes

- Started from pilot checkpoint, target 240 million seen tokens.
- Step 6000: 49.15 million seen tokens; old English-first validation loss 2.545. The same checkpoint measured 3.792 with evenly spaced full-file windows.
- Step 7000: 57.34 million seen tokens; English loss 2.211, Chinese loss 4.551, 65/35 weighted loss 3.030. This is the baseline before changing language sampling.
- From step 7000 onward: training batches sample English with probability 0.74, Chinese with probability 0.26. The cumulative target is approximately 65/35 over 240 million seen tokens.
- Step 7500: English loss 2.113, Chinese loss 4.646, weighted loss 3.000.
- Step 8000: English loss 2.061, Chinese loss 4.681, weighted loss 2.978. Chinese loss has not improved yet; continue measuring instead of interpreting only the weighted number.
- Step 10000: 81.92 million seen tokens; English loss 1.953, Chinese loss 4.665, weighted loss 2.902. GPU utilization was about 96%, with about 2.1 GB GPU memory used.
- Fixed prompt `Once upon a time` at step 10000: produces grammatical sentences and dialogue, but repeats the name Lily and contradicts who is speaking.
- Fixed prompt `从前有一个` at step 10000: produces Chinese words and Wikipedia-like phrasing, but drifts into incoherent historical claims. This matches the much higher Chinese validation loss.
- The short pilot's cosine schedule reached its minimum LR near step 611. Extending the same checkpoint to a 240 million token target changed the schedule and raised LR near the maximum at step 623. This is a real training-stage discontinuity; later interpretation should not treat the pilot and long run as one unchanged LR schedule.

## Final result

- Completed at step 29,297 with 240,001,024 seen tokens. The final checkpoint is `checkpoints/final_step_00029297.pt`.
- Last 50-batch validation point at step 29,250: English loss 1.65585, Chinese loss 4.37700, 65/35 weighted loss 2.60825.
- Independent final evaluation with 100 fixed batches per language: English loss 1.64872, Chinese loss 4.38744, 65/35 weighted loss 2.60727, weighted perplexity 13.56201.
- The weighted loss fell from 3.016 at step 7,250 to 2.608 at step 29,250. Over the last 4,250 steps it fell only from 2.629 to 2.608. This is a clear slowdown, but not proof of complete convergence.
- Final English sample from `Once upon a time`: a dinosaur helps Lily find a toy, then the narrative becomes contradictory. Sentence structure and local coherence improved from the pilot.
- Final Chinese sample from `从前有一个`: produces locally plausible phrases about a child and research, but repeats concepts and loses coherent meaning. Some rare characters may become unknown tokens because this tokenizer has no byte fallback.
- The final model is a learning and assessment artifact, not a general-purpose assistant. It has not undergone supervised instruction fine-tuning or alignment.
