from reranker import get_reranked_texts
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline
from textwrap import dedent
from transformers import BartForConditionalGeneration, BartTokenizer


def get_final_answers(query):
    answers=get_reranked_texts(query=query,k=3)

    text=f"{answers[0]}"

    for i in range(1,3):
        text=f"{text}. {answers[i]}"

    data_row={
        'question': query,
        'context': text
    }

    generator = pipeline("text-generation",max_new_tokens=100, model="gpt2")


    response = generator(text,num_return_sequences=1)
    # print(text)
    # print('---------------------------')
    # print(response[0]['generated_text'])
    return response[0]['generated_text']


print(get_final_answers("what is a queue"))