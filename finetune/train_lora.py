"""LoRA fine-tune whisper-large-v3 on the Flow corpus.

Scaffold — run only when the corpus is larger and a WER benchmark shows a real
acoustic gap (see README). Not wired into Flow's runtime.

Deps:  pip install transformers datasets peft accelerate librosa evaluate soundfile
Run:   python finetune/train_lora.py
Then:  ct2-transformers-converter --model finetune/out/merged \
         --output_dir models/flow-v3-ct2 --quantization float16
       # config.json -> "model": "C:/Calude/flow/models/flow-v3-ct2"
"""

import json
from pathlib import Path

BASE = "openai/whisper-large-v3"
HERE = Path(__file__).resolve().parent
TRAIN = HERE / "dataset" / "train.jsonl"
EVAL = HERE / "dataset" / "eval.jsonl"
OUT = HERE / "out"


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    import evaluate
    import librosa
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    processor = WhisperProcessor.from_pretrained(BASE, language="en", task="transcribe")

    def prep(row):
        audio, _ = librosa.load(row["audio"], sr=16000)
        row["input_features"] = processor.feature_extractor(
            audio, sampling_rate=16000).input_features[0]
        row["labels"] = processor.tokenizer(row["text"]).input_ids
        return row

    train = Dataset.from_list(load_jsonl(TRAIN)).map(prep, remove_columns=["audio", "text"])
    evald = Dataset.from_list(load_jsonl(EVAL)).map(prep, remove_columns=["audio", "text"])

    model = WhisperForConditionalGeneration.from_pretrained(BASE)
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none"))
    model.print_trainable_parameters()

    import torch

    def collate(batch):
        feats = [{"input_features": b["input_features"]} for b in batch]
        labs = [{"input_ids": b["labels"]} for b in batch]
        b1 = processor.feature_extractor.pad(feats, return_tensors="pt")
        b2 = processor.tokenizer.pad(labs, return_tensors="pt")
        labels = b2["input_ids"].masked_fill(b2.attention_mask.ne(1), -100)
        b1["labels"] = labels
        return b1

    wer = evaluate.load("wer")

    def metrics(pred):
        pred_ids, label_ids = pred.predictions, pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        h = processor.batch_decode(pred_ids, skip_special_tokens=True)
        r = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * wer.compute(predictions=h, references=r)}

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUT), per_device_train_batch_size=8,
        gradient_accumulation_steps=2, learning_rate=1e-4, warmup_steps=50,
        num_train_epochs=3, fp16=True, eval_strategy="epoch",
        predict_with_generate=True, generation_max_length=225,
        save_strategy="epoch", logging_steps=10, report_to=[])

    Seq2SeqTrainer(model=model, args=args, train_dataset=train,
                   eval_dataset=evald, data_collator=collate,
                   compute_metrics=metrics,
                   tokenizer=processor.feature_extractor).train()

    model.merge_and_unload().save_pretrained(OUT / "merged")
    processor.save_pretrained(OUT / "merged")
    print(f"done -> {OUT/'merged'}  (convert to CT2 next; see README)")


if __name__ == "__main__":
    main()
