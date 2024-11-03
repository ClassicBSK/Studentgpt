from sentence_transformers import SentenceTransformer

from preprocess2 import get_final_data

finaldata=get_final_data()
# print(final_data)

model=SentenceTransformer('models\models--sentence-transformers--all-MiniLM-L6-v2\embed')



def get_embeddingsv2(query:str):
    tempfinal=finaldata.copy()
    tempfinal.append(query)
    
    tokenized_texts = [text.lower().split() for text in tempfinal]
    embeddings=model.encode(sentences=tempfinal)
    return embeddings
