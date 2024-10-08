from transformers import T5TokenizerFast,T5ForConditionalGeneration
import os

class config:
    num_workers=1
    n_epochs=3
    batch_size=8

    text_max_len=512
    sum_max_len=128

    learning_rate=0.0001

    t5_tokenizer=T5TokenizerFast.from_pretrained("t5-base")

    t5_pretrained_model=T5ForConditionalGeneration.from_pretrained("t5-base",return_dict=True)
