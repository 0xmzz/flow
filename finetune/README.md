# Fine-tuning Flow to your voice

## TL;DR — recommendation

**Don't full-fine-tune yet.** The corpus analysis shows your corrections are
**71% formatting** (articles, `gonna→going to`, `front end→frontend`) which the
LLM cleanup layer already handles, and only **29% acoustic mishears** — mostly
proper nouns (`clod/cloud→claude`, `monohydride→monohydrate`) that **vocabulary
biasing fixes for free, no training**. Whisper `large-v3`'s acoustic accuracy on
your voice is already high.

Highest-leverage order:

1. **Vocab biasing (done, zero cost)** — acoustic mishears added to `config.json`
   `vocab`. Add new names/tickers as you notice misses.
2. **Keep collecting.** Flow now auto-archives every dictation + audio. Revisit
   fine-tuning at **10+ hours** (we have 2.5) — small corpora overfit.
3. **LoRA fine-tune (scaffold below), only if** a WER benchmark on held-out
   audio shows a real acoustic gap. Full fine-tuning of large-v3 on 2.5h would
   overfit for marginal gain.

## The dataset (built by `prepare_dataset.py`)

From `data/imported-wispr/` (1072 Wispr dictations + audio):

- **622 usable clips, 2.47 hours**, all 16 kHz mono
- **205 corrected** (label = your edit), 417 accepted (label = ASR, i.e. correct)
- Split: `train.jsonl` 581 · `eval.jsonl` 41 — `{audio, text}` per line
- `corrections.json` — asr→edit substitutions (mine for vocab)
- Label rule: `editedText` if you fixed it, else `asrText`. `formattedText` is
  excluded — it's Wispr's LLM output, not a verbatim spoken-word target.

Regenerate anytime: `python finetune/prepare_dataset.py`.

## If/when you fine-tune: the pipeline

faster-whisper runs CTranslate2 (inference-only), so you fine-tune the PyTorch
model then convert back:

1. **LoRA fine-tune** `openai/whisper-large-v3` on `train.jsonl`
   (`train_lora.py`). LoRA = parameter-efficient, overfit-resistant on small data.
   ~30–60 min on the 5090; needs `pip install transformers datasets peft
   accelerate librosa evaluate`.
2. **Merge + convert to CT2**:
   `ct2-transformers-converter --model <merged> --output_dir models/flow-v3-ct2
   --quantization float16`
3. Point Flow at it: `config.json` → `"model": "C:/Calude/flow/models/flow-v3-ct2"`.
   faster-whisper loads a local CT2 dir directly.

## Honest scope

- **Expected gain:** modest. Accent adaptation from 2.5h helps a little; the
  proper-noun wins are already captured by biasing. Don't expect a night-and-day
  change — the data says there isn't one to be had yet.
- **Risk:** overfitting to 622 clips (repeated phrasing, one mic, one room) can
  make it *worse* on novel input. LoRA + a held-out eval mitigate this.
- **The real payoff is later:** the corpus grows daily now. At 10–20h across
  more contexts, a LoRA genuinely adapts large-v3 to your accent + domain in a
  way no cloud dictation app can match — because they don't give you your data,
  and you have it.
