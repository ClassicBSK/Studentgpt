# from transformers import BartForConditionalGeneration, BartTokenizer
from retrieval3 import get_answer
# # model_name = 'facebook/bart-large-cnn'
# model = BartForConditionalGeneration.from_pretrained('Yale-LILY/brio-cnndm-uncased')
# tokenizer = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-uncased')

# def summarize_article(article):
#     # Load BART model and tokenizer

#     # Tokenize and encode the article
#     inputs = tokenizer.encode(article, return_tensors='pt',
# max_length=1024, truncation=True)

#     # Generate summary
#     summary_ids = model.generate(inputs, num_beams=4, max_length=1024,
# early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     return summary

query='what are the operations of a stack'

def get_reranked_texts(query,k):
    text=get_answer(query=query,k=6)[0]

    from sentence_transformers import CrossEncoder

    model = CrossEncoder("models\models--jinaai--jina-reranker-v1-turbo-en\\reranker", trust_remote_code=True)

    results = model.rank(query, text, return_documents=True, top_k=k)

    lise=[]
    for i in results:
        lise.append(i['text'])
    return lise

