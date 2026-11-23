import torch, evaluate
from PIL import Image
from transformers import (TrOCRProcessor, VisionEncoderDecoderModel,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          default_data_collator)

from convert_jsonl_content_to_tuple import convert_jsonl_content_to_tuple 
from pathlib import Path
import json,glob,re,os
import pandas as pd
import matplotlib.pyplot as plt
import file_manipulations

CKPT = "microsoft/trocr-small-handwritten"
processor = TrOCRProcessor.from_pretrained(CKPT)
model = VisionEncoderDecoderModel.from_pretrained(CKPT)

# the settings people most often get wrong
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.generation_config.max_length = 64        # check against your longest label
model.generation_config.num_beams = 4
model.generation_config.early_stopping = True

class LineDS(torch.utils.data.Dataset):
    def __init__(self, rows, root, max_len=64):   # rows: [(filename, text), ...]
        self.rows, self.root, self.max_len = rows, root, max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        fn, text = self.rows[i]
        img = Image.open(f"{self.root}/{fn}").convert("RGB")
        pv = processor(img, return_tensors="pt").pixel_values[0]
        ids = processor.tokenizer(text, padding="max_length", truncation=True,
                                  max_length=self.max_len).input_ids
        ids = [t if t != processor.tokenizer.pad_token_id else -100 for t in ids]
        return {"pixel_values": pv, "labels": torch.tensor(ids)}

cer = evaluate.load("cer")
def metrics(p):
    pred = processor.batch_decode(p.predictions, skip_special_tokens=True)
    p.label_ids[p.label_ids == -100] = processor.tokenizer.pad_token_id
    ref = processor.batch_decode(p.label_ids, skip_special_tokens=True)
    return {"cer": cer.compute(predictions=pred, references=ref)}

args = Seq2SeqTrainingArguments(
    output_dir="trocr-ft", predict_with_generate=True,
    per_device_train_batch_size=8, per_device_eval_batch_size=8,
    learning_rate=2e-5, num_train_epochs=3,warmup_steps=0.05,logging_strategy="steps",logging_steps=10,eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="cer", greater_is_better=False, fp16=False)

percentages = [0.25,0.50,1.0]
n = 270
folder_val = Path("current_samples_folder")
folder_val_mkdir(parents=True,exist_ok=True)
root = "trocr-ft"

for p in percentages:
    summary = []
    num = round(p*n)
    file_manipulations.clear_folder("current_samples_folder")
    file_manipulations.copy_first_n_percent_files("syn_scans_batch_1/train/images","current_samples_folder/train/images",num)
    file_manipulations.copy_first_n_percent_files("syn_scans_batch_1/val/images","current_samples_folder/val/images",num)
    file_manipulations.copy_n_file_lines("syn_scans_batch_1/train/train.jsonl","current_samples_folder/train/train.jsonl",num) 
    file_manipulations.copy_n_file_lines("syn_scans_batch_1/val/val.jsonl","current_samples_folder/val/val.jsonl",num)
    file_manipulations.delete_n_file_lines("current_samples_folder/train/train.jsonl",num)
    file_manipulations.delete_n_file_lines("current_samples_folder/val/val.jsonl",num)

    train_rows = convert_jsonl_content_to_tuple("current_samples_folder/train/train.jsonl")
    train_dataset = LineDS(train_rows,"current_samples_folder/train/images",len(train_rows))
    eval_rows = convert_jsonl_content_to_tuple("current_samples_folder/val/val.jsonl")
    eval_dataset = LineDS(eval_rows, "current_samples_folder/val/images", len(eval_rows))
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, compute_metrics=metrics,
                         data_collator=default_data_collator)
    trainer.train()

    for path in glob.glob(os.path.join(root,"checkpoint-*", "trainer_state.json")):
        with open(path) as f:
            s = json.load(f)
        ckpt_step = int(re.search(r"checkpoint-(-\d+)", path).group(1))
        summary.append({
            "checkpoint": ckpt_step,
            "global_step": s["global_step"],
            "epoch": s["epoch"],
            "best_metric": s.get("best_metric"),
            "best_model_checkpoint": s.get("best_model_checkpoint"),
        })
    summary = pd.DataFrame(summary).sort_values("checkpoint")
    print(summary)

    latest = max(glob.glob(os.path.join(root, "checkpoint-*")),
        key = lambda p: int(p.split("-")[-1]))
    with open(os.path.join(latest, "trainer_state.json")) as f:
        log = pd.DataFrame(json.load(f)["log_history"])
    train_log = log[log["loss"].notna()] if "loss" in log else pd.DataFrame()
    eval_log = log[log["eval_loss"].notna()] if "eval_loss" in log else pd.DataFrame()
    print(eval_log[["step", "epoch", "eval_loss", "eval_cer"]])
   
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
   
    ax[0].plot(train_log["step"], train_log["loss"], label="train_loss")
    ax[0].plot(eval_log["step"], eval_log["eval_loss"], marker = "o", label="eval loss")
    ax[0].set_xlabel("step"); ax[0].legend(); ax[0].set_title("Loss")

    ax[1].plot(eval_log["step"], eval_log["eval_cer"], marker = "o")
    ax[1].set_xlabel("step"); ax[1].set_title("Eval CER")

    plt.tight_layout(); plt.show()

    
