import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
import pickle

nltk.download('punkt')
nltk.download('stopwords')


file=open("Text\data.txt",mode="r",encoding="utf-8")

#splitting based on tokens
text1=file.read()
text=text1.split('<START>')
text=[i.lower() for i in text]
text=text[1:]
finaldata=[]

#removing new lines
for i in text:
    i=i.replace('\n',' ')
    finaldata.append(i)

#checked tokens

stopwords = nltk.corpus.stopwords.words('english')
def tokenize(tex):
    tokens=[]
    text=[tex]
    for i in text:
        for word in nltk.word_tokenize(i):
            tokens.append(word)
    #print(tokens)
    return tokens

#tokenize(text[1])

def stem_tokenize(text):
    tokens=tokenize(text)
    stemmer = SnowballStemmer('english')
    stemmed_tokens=[stemmer.stem(word) for word in tokens]
    #print(stemmed_tokens)
    return stemmed_tokens

#stem_tokenize(text[2])

def vocab_table(text):
    vocab_tokenized=[]
    vocab_stemmed=[]
    total_words=[]
    for word in text:
        sentence=tokenize(word)
        total_words.append(sentence)
        vocab_tokenized.extend(sentence)
        stementence=stem_tokenize(word)
        vocab_stemmed.extend(stementence)
    #print(vocab_stemmed)
    return vocab_tokenized,vocab_stemmed,total_words

vocab_tokenized,vocab_stemmed,total_words=vocab_table(finaldata)

import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

def get_embeddings(query:str):
    tempfinal=finaldata.copy()
    tempfinal.append(query)
    
    tokenized_texts = [text.lower().split() for text in tempfinal]
    model = Word2Vec(sentences=tokenized_texts, window=8, min_count=1, workers=50)
    embeddings = np.array([np.mean([model.wv[word] for word in text if word in model.wv], axis=0) for text in tokenized_texts])
    
    return embeddings

def get_final_data():
    return finaldata
'''embeddings=get_embeddings("hello ")



# Dimensionality reduction to 3D using PCA
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)
lise=[]
for i in range(len(finaldata)):
    print(i)
    lise.append(i)
    print('--------------------')
    print(finaldata[i])

# Visualize the 3D embeddings
fig = px.scatter_3d(x=embeddings_3d[:,0], y=embeddings_3d[:,1], z=embeddings_3d[:,2],
                    text=lise, title="Text Similarity in 3D")

fig.show()
'''

