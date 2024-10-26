from preprocess2 import get_embeddings,get_final_data
from embed2 import get_embeddingsv2
import numpy as np
from numpy.linalg import norm
import plotly.express as px
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')

stopwlist=stopwords.words('english')
#query="ways to implement a stack"

finaldata=get_final_data()

#print(embeddings)
def show_3d():
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

def cosine_similarity(embed1,embed2):
    ret=np.dot(embed1,embed2)/(norm(embed1)*norm(embed2))
    return ret

def get_scores(query:str,k:int):
    embeddings=get_embeddingsv2(query=query)
    scoresdict={}
    for i in range(0,len(embeddings)-1):
        scoresdict[i]=cosine_similarity(embeddings[i],embeddings[len(embeddings)-1])
        
        # print(sorteddict)
    sorteddict=sorted(scoresdict.items(),key= lambda x:x[1])
    sorteddict=sorteddict[::-1]
    sorteddict=sorteddict[:k]
    lie=[]
    # print(sorteddict)
    for i in sorteddict:
        lie.append(i[0])
    return lie

def get_set(query:str,k:int):
    tempsite=get_scores(query=query,k=k)
    lise=query.split()
    for i in lise:
        if i  not in stopwlist:
            #print(i)
            tempsite.extend(get_scores(query=i,k=k))
    tempsite=list(set(tempsite))
    print(tempsite)
    return tempsite

def get_final_sentences(query:str,k:int):
    seti=get_set(query=query,k=k)
    corpus=[]
    for i in seti:
        corpus.append(finaldata[i])
    
    return corpus


#show_3d("what is a stack?")
#lite=get_set("what is a stack?",3)
#print(lite)
