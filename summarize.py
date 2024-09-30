from transformers import PegasusForConditionalGeneration, AutoTokenizer
import torch

# You can chose models from following list
# https://huggingface.co/models?sort=downloads&search=google%2Fpegasus
model_name = 'google/pegasus-cnn_dailymail'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
batch = tokenizer(txt, truncation=True, padding='longest', return_tensors="pt").to(device)
translated = model.generate(**batch)
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

tgt_text