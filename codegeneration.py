from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from preprocess2 import get_paratoken

topics=['Implement a queue','Implement a stack','Implement a linked list','Implement a binary tree','Implement a graph']
def get_topic(text):
    paratoken=get_paratoken()
    return paratoken[text]

def generate_code(tex):
    topic=get_topic(tex)
    text=topics[topic]
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi")

    input_ids = tokenizer(text, return_tensors="pt").input_ids

    generated_ids = model.generate(input_ids, max_length=1000)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

