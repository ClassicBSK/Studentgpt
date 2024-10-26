from retrieval import get_final_sentences
from fastbm25 import fastbm25
import nltk

from nltk.stem.snowball import SnowballStemmer

#from preprocessing import *

query="what is a stack?"



def get_answer(query:str,k:int):
    stemmer = SnowballStemmer('english')
    corpus=get_final_sentences(query=query,k=k)
    tokenized_corpus=[i.lower().split() for i in corpus]
    print(tokenized_corpus)
    model=fastbm25(tokenized_corpus)

    result=model.top_k_sentence(query,k=1)
    return result

#print(get_answer(query=query,k=3)[0][0])
get_answer(query=query,k=3)[0][0]