# Debug Log

This file records real problems observed during the run. Do not add an issue until it has actually happened.

> 本日志同时保留旧中英方案与新英文单语方案的问题。2026-11-10 的条目属于旧方案；
> 2026-11-11 的英文训练恢复与英文 Task B 提示词问题属于当前方案。
> 这些问题由 Codex 协助发现、定位或修复，不应表述为使用者独立完成。

## Entry template

- Date/time:
- Command or stage:
- Symptom:
- First measurement:
- Hypotheses:
- Checks performed:
- Root cause:
- Fix:
- Evidence after the fix:
- What I would check first next time:

## 2026-11-10 SentencePiece output path mismatch

- Date/time: 2026-11-10
- Command or stage: First corpus preparation run, after SentencePiece training
- Symptom: Tokenizer training completed, but the encoding stage could not find `tokenizer/spm.model`.
- First measurement: The directory contained `tokenizer/spm.model.model` and `tokenizer/spm.model.vocab`.
- Hypotheses: The tokenizer training failed, the current directory was wrong, or the model prefix was interpreted differently than expected.
- Checks performed: Inspected the SentencePiece log and compared its reported output path with the configured path.
- Root cause: SentencePiece appends `.model` and `.vocab` to `model_prefix`; passing a path already ending in `.model` created an extra suffix.
- Fix: Normalize a `.model` input path to a suffix-free `model_prefix` before calling SentencePiece.
- Evidence after the fix: The second run created `tokenizer/spm.model` and `tokenizer/spm.vocab`, then encoded 112,968,440 training tokens and 2,401,669 validation tokens successfully.
- What I would check first next time: Read the library's reported output filenames before starting a long corpus encoding pass.

## 2026-11-10 Automatic checkpoint discovery

- Date/time: 2026-11-10
- Command or stage: Resume test after the 5 million token pilot
- Symptom: The first implementation of `--resume auto` searched only `step_*.pt`, so a run ending before the first periodic checkpoint would not be found.
- First measurement: The pilot produced `final_step_00000611.pt`, while automatic discovery returned no checkpoint.
- Hypotheses: The checkpoint was written to the wrong directory, the filename pattern was too narrow, or the checkpoint itself was corrupt.
- Checks performed: Listed `checkpoints/`, inspected the filename, and compared it with the glob used by `latest_checkpoint`.
- Root cause: Final and interrupt checkpoints have prefixes before `step_`, but the glob assumed the filename started with `step_`.
- Fix: Search all `*_step_*.pt` files and sort by the numeric step suffix.
- Evidence after the fix: A short continuation resumed from `final_step_00000611.pt` and finished at step 623.
- What I would check first next time: Test auto-resume immediately after every new checkpoint naming convention.

## 2026-11-10 Validation sampling bias

- Date/time: 2026-11-10
- Command or stage: Long training, after step 5000
- Symptom: The validation loss looked much lower than typical training batches.
- First measurement: At step 6000 the original validation loss was about 2.55, while a fixed set of windows spread across the full validation file gave 3.79 from the same checkpoint.
- Hypotheses: The training distribution differed from the validation distribution, or the evaluator was reading a narrow part of the validation file.
- Checks performed: Inspected the corpus writer and evaluator. The writer placed English documents before Chinese documents; the evaluator took only the first 50 batches.
- Root cause: Validation sampled only the beginning of the file, which is English TinyStories. It did not represent the bilingual validation corpus.
- Fix: Use deterministic, evenly spaced windows across the full validation token file. Label new validation records with `val_sampling=evenly_spaced_full_corpus_v2`.
- Evidence after the fix: The step 6000 checkpoint measured 3.79 loss under the new full-corpus evaluation, compared with about 2.55 under the old English-first method.
- What I would check first next time: Verify the validation sample indices and language coverage before interpreting the first loss curve.

## 2026-11-10 Character quotas did not produce the intended language mix

- Date/time: 2026-11-10
- Command or stage: Corpus audit during long training
- Symptom: The prepared training corpus did not meet the planned 65% English / 35% Chinese token ratio.
- First measurement: `data_stats.py` counted 42,420,185 English tokens and 70,548,255 Chinese tokens in the training file, or 37.6% / 62.4%.
- Hypotheses: The source sizes were wrong, or English and Chinese had different characters-per-token ratios.
- Checks performed: Counted actual SentencePiece tokens separately for the contiguous English and Chinese parts of both training and validation text.
- Root cause: The data limits were set in characters (180 million English, 100 million Chinese), but Chinese required many more tokens per character than English under the joint tokenizer.
- Fix: From step 7000, sample a language first, using 74% English, then choose a random token window inside that language's span. This compensates for the earlier 57 million tokens sampled at the original ratio. Evaluate English and Chinese separately and report a 65% / 35% weighted loss.
- Evidence after the fix: The sampled dataset and language-specific validation loaders returned correct shapes and spans. The step 7000 checkpoint measured English loss 2.2113, Chinese loss 4.5508, weighted loss 3.0301 before the sampling change. Later training measurements will show whether the language gap narrows.
- What I would check first next time: Count tokens by language immediately after tokenization, before starting a long run.

## 2026-11-11 English single-language resume rejected CUDA RNG state

- Date/time: 2026-11-11
- Command or stage: Resume English single-language base training from `checkpoints/english_single/base/step_00006000.pt`
- Symptom: Training loaded the model configuration and checkpoint metadata, then failed before the first resumed batch with `TypeError: RNG state must be a torch.ByteTensor`.
- First measurement: The exception came from `torch.cuda.set_rng_state_all`; the checkpoint contained `cuda_rng_state`, and the loader had used `map_location=cuda`.
- Hypotheses: The checkpoint was corrupt, the CUDA RNG list had the wrong number of devices, or mapping the state to CUDA changed the tensor type/device expected by the restore API.
- Checks performed: Confirmed the checkpoint file existed and was the new English checkpoint; inspected `torch.cuda.get_rng_state_all()` and the load path. The saved values are byte tensors, but `map_location=cuda` moves them to CUDA before restore.
- Root cause: `torch.cuda.set_rng_state_all` requires CPU `torch.ByteTensor` objects, while the resume loader passed the mapped CUDA tensors directly.
- Fix: Convert every saved CUDA RNG tensor back to CPU before calling `set_rng_state_all`.
- Evidence after the fix: The English run resumed from step 6000 and completed at step 29297; the failed attempt did not update model or optimizer state.
- What I would check first next time: Test checkpoint restore with both CPU and CUDA RNG states after any change to `map_location`.

## 2026-11-11 English Task B prompt selector assumed Chinese text

- Date/time: 2026-11-11
- Command or stage: First launch of the new English-only Task B
- Symptom: The run stopped before creating a checkpoint with `RuntimeError: Not enough long Chinese documents in validation text`.
- First measurement: The new validation file contains only TinyStories English text, while `select_prompts()` filtered every line unless it contained Chinese characters.
- Hypotheses: The validation file was empty or too short, the tokenizer produced too few tokens, or the prompt selector still encoded the old bilingual assumption.
- Checks performed: Counted the new validation tokens and inspected the selector. The file contains 842,751 English tokens; the selector hard-coded a Chinese Unicode range and a Chinese-specific error.
- Root cause: The old Task B qualitative generation helper was reused without a language-neutral branch.
- Fix: Select Chinese documents when a validation file contains Chinese; otherwise select sufficiently long English documents and use a generic error message.
- Evidence after the fix: The code now supports both the frozen bilingual Task B and the new English-only Task B; this first failed launch created no new checkpoint.
- What I would check first next time: Run language and prompt-length assertions before starting a long experiment.
