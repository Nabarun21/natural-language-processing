import nltk
import pickle
import re
import numpy as np
import joblib
nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.jlib',
    'TAG_CLASSIFIER': 'tag_classifier.jlib',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.jlib',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    
    ret_embeddings={}    
    
    for line in open(embeddings_path, encoding='utf-8'):
        word,*emb=line.strip().split('\t')
        ret_embeddings[word] = [np.float32(x) for x in emb]
    dim=len(ret_embeddings[list(ret_embeddings.keys())[0]])
    return ret_embeddings,dim


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    
    ret_val=np.zeros(dim)
    question=question.strip()
    words=question.split(" ")
    counter=0
    for word in words:
        if word in embeddings:
            counter+=1
            ret_val+=embeddings[word]
       
    if counter==0:counter+=1
    return ret_val/counter
    
    

def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return joblib.load(f)
