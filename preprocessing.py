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

text.append("what is a stack?")
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
def tfidfy(text):
    tfid_vectorizer=TfidfVectorizer(max_df=0.85,min_df=0.1,sublinear_tf=True,stop_words='english',use_idf=True,tokenizer=tokenize,ngram_range=(1,10),token_pattern=None)
    tfid_matrix=tfid_vectorizer.fit(text)
    tfid_matrix=tfid_vectorizer.transform(text)
    return tfid_matrix

tfid_matrix=tfidfy(text)
print(tfid_matrix.get_shape())

#kmeans try
'''
kmeans=[KMeans(n_clusters=i,n_init=100, max_iter=500) for i in range(1,14)]
score=[kmeans[i].fit(tfid_matrix).score(tfid_matrix) for i in range(len(kmeans))]

nc=range(1,14)
plt.plot(nc,score)
plt.xlabel('no. of clusters')
plt.show()
'''

#kmeans=KMeans(n_clusters=9,n_init=2000,max_iter=6000)

with open("model.pkl","rb") as f:
    kmeans=pickle.load(f)

'''
kmeans.fit(tfid_matrix)
with open("model.pkl", "wb") as f:
    pickle.dump(kmeans, f) 
'''
clusters=kmeans.predict(tfid_matrix)
clusters=list(clusters)
print(clusters)
colour=['red','blue','green','grey','orange','violet','black','brown','beige','yellow']
for i in range(len(clusters)):
    print(i)
    print(text[i])
    print('----------------------------------------')
    #if(clusters[i]==4):
    #   print('true')
    plt.scatter(clusters[i],i,color=colour[clusters[i]])


plt.show()

'''
with open("model.pkl", "wb") as f:
    pickle.dump(kmeans, f) 
    '''  
