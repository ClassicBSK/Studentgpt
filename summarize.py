from reranker import get_reranked_texts
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline
from textwrap import dedent
from transformers import BartForConditionalGeneration, BartTokenizer
import torch




def get_final_answers(query,mode=None|str):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    answers=get_reranked_texts(query=query,k=3)

    text=f"{answers[0]}"

    for i in range(1,3):
        text=f"{text}. {answers[i]}"

    data_row={
        'question': query,
        'context': text
    }
    try:
        generator = pipeline("text-generation",max_new_tokens=20, model="models\\models--gpt2\\gpt2")
        response = generator(text,num_return_sequences=1)
        # print(text)
        # print('---------------------------')
        # print(response[0]['generated_text'])
        return response[0]['generated_text']
    except:
        return text

    


# print(get_final_answers("applications of a linked list"))