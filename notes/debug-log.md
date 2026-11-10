# Debug Log

This file records real problems observed during the run. Do not add an issue until it has actually happened.

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
