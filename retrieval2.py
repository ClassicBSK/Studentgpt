from retrieval import get_final_sentences
from fastbm25 import fastbm25
#from preprocessing import *

query="main stack operations"

def get_answer(query:str,k:int):
    corpus=get_final_sentences(query=query,k=k)
    tokenized_corpus=[i.lower().split() for i in corpus]
    model=fastbm25(tokenized_corpus)

    result=model.top_k_sentence(query,k=1)
    return result

print(get_answer(query=query,k=3))