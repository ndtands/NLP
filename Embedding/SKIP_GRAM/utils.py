import numpy as np
import string
from nltk.corpus import stopwords

def softmax(x):
    """ Compute softmax"""
    e_x = np.exp(x-np.max(x))
    return e_x/e_x.sum()

def preprocessing(corpus):
    """  
    - split text to words
    - remove stopword
    - remove punctuation
    """
    stop_words = set(stopwords.words('english'))   
    training_data = []
    sentences = corpus.split(".")
    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip()
        sentence = sentences[i].split()
        x = [word.strip(string.punctuation) for word in sentence
                                     if word not in stop_words]
        x = [word.lower() for word in x]
        training_data.append(x)
    return training_data
      
  
def prepare_data_for_training(sentences,w2v):
    data = {}
    for sentence in sentences:
        for word in sentence:
            if word not in data:
                data[word] = 1
            else:
                data[word] += 1
    V = len(data)
    #sort data flow comom word
    #vocab is word2idx
    data = sorted(list(data.keys()))
    vocab = {}
    for i in range(len(data)):
        vocab[data[i]] = i
    for sentence in sentences:
        for i in range(len(sentence)):
            center_word = [0 for x in range(V)]
            center_word[vocab[sentence[i]]] = 1
            context = [0 for x in range(V)]
             
            for j in range(i-w2v.window_size,i+w2v.window_size):
                if i!=j and j>=0 and j<len(sentence):
                    context[vocab[sentence[j]]] += 1
            w2v.X_train.append(center_word)
            w2v.y_train.append(context)

    #center_word: one hot.
    #content: more than 1 number one.
    w2v.initialize(V,data)
  
    #return w2v.X_train,w2v.y_train  