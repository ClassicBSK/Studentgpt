from reranker import get_reranked_texts
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline
from textwrap import dedent
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from codegeneration import generate_code


def get_final_answers(query,mode=None|str):
    answers=get_reranked_texts(query=query,k=3)

    text=f"{answers[0]}"
    # code=generate_code(answers[0])
    # print(code)
    print('-------------------------')
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
        # print('-------------------------')
        return response[0]['generated_text']
    except:
        return text

    


# print(get_final_answers("what is a queue"))