import torch, evaluate
from PIL import Image
from transformers import (TrOCRProcessor, VisionEncoderDecoderModel,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          default_data_collator)

from convert_jsonl_content_to_tuple import convert_jsonl_content_to_tuple 

CKPT = "microsoft/trocr-base-handwritten"
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
    learning_rate=2e-5, num_train_epochs=15, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="cer", greater_is_better=False, fp16=True)

train_rows = convert_json_content_to_tuple("handwritten_ocr/train.jsonl")
train_dataset = LineDS(train_rows,"handwritten_ocr/train/images",len(train_rows))
eval_rows = convert_json_content_to_tuple("handwritten_ocr/val.jsonl")
eval_dataset = LineDS(eval_rows, "handwritten_ocr/val/images", len(eval_rows))
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, compute_metrics=metrics,
                         data_collator=default_data_collator)
trainer.train()
