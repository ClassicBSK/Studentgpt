import bm25s
from retrieval import get_final_sentences

import snowballstemmer




query="how is a stack implemented using a simple array"

def get_answer(query:str,k:int):
    stemmer = snowballstemmer.stemmer('english')
    corpus=get_final_sentences(query=query,k=k)
    tokenized_corpus=bm25s.tokenize(corpus,stopwords="en",stemmer=stemmer)
    # print(tokenized_corpus)

    retriever=bm25s.BM25()
    retriever.index(tokenized_corpus)

    tokenized_query=bm25s.tokenize(query,stemmer=stemmer)

    results,score=retriever.retrieve(tokenized_query,corpus=corpus,k=k)

    print(results)

    
    return results

get_answer(query=query,k=3)

