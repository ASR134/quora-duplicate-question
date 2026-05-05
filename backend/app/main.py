from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel, Field
import os
from typing import Annotated
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix, hstack
from nltk.stem import PorterStemmer
import distance
from fastapi.responses import JSONResponse
import nltk
nltk.download('punkt')
nltk.download('stopwords')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
exclude = string.punctuation

# paths
model_path = os.path.join(BASE_DIR,"model", "quora_model.pkl")
tf_path = os.path.join(BASE_DIR,"model","quora_tf.pkl")
scalar_path = os.path.join(BASE_DIR,"model","quora_scalar.pkl")


# load models
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(tf_path, "rb") as f:
    tf = pickle.load(f)

with open(scalar_path, "rb") as f:
    scalar = pickle.load(f)


app = FastAPI()


# pydantic model
class Question_input(BaseModel):
    
    question1 : Annotated[str,Field(...,description="First Question")]
    question2 : Annotated[str,Field(...,description="Second Question")]


# TEXT PREPROCESSING
def remove_punc(text):
    return text.translate(str.maketrans('','',exclude))

def remove_html_tags(text):
    clean = re.sub(r'<.*?>', '', text)
    return clean

def pre_process(q):

    q = q.lower().strip()

    q = q.replace('[math]','')
    q = q.replace('%', 'percent ')
    q = q.replace('$', 'dollar ')
    q = q.replace('₹', 'rupee ')
    q = q.replace('€', 'euro ')
    q = q.replace('@', 'at ')

    q = q.replace(',000,000,000', 'b ')
    q = q.replace(',000,000', 'm ')
    q = q.replace(',000', 'k ')
    q = re.sub(r'([0-9]+)000000000',r'\1b',q)
    q = re.sub(r'([0-9]+)000000',r'\1b',q)
    q = re.sub(r'([0-9]+)000',r'\1k',q)

    contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",

    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",

    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",

    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",

    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",

    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",

    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",

    "let's": "let us",

    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",

    "needn't": "need not",
    "needn't've": "need not have",

    "o'clock": "of the clock",

    "oughtn't": "ought not",
    "oughtn't've": "ought not have",

    "shan't": "shall not",
    "sha'n't": "shall not",

    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",

    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",

    "so've": "so have",
    "so's": "so is",

    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",

    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",

    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",

    "to've": "to have",

    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",

    "weren't": "were not",

    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",

    "when's": "when is",
    "when've": "when have",

    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",

    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",

    "why's": "why is",
    "why've": "why have",

    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",

    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",

    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",

    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    l = []
    for word in q.split():
        if word in contractions:
            l.append(contractions[word])
        else:
            l.append(word)
    
    q = " ".join(l)
    q = q.replace("'ve"," have")
    q = q.replace("n't"," not")
    q = q.replace("'re"," are")
    q = q.replace("'ll"," will")

    q = remove_html_tags(q)

    q = remove_punc(q)

    #stemming
    ps = PorterStemmer()
    q = " ".join([ps.stem(word) for word in word_tokenize(q)])

    return q



# BASIC FEATURE EXTRACTION
def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(),row['question1'].split(" "))) # map(function,iterable) - passes each element of iterable to function
    w2 = set(map(lambda word: word.lower().strip(),row['question2'].split(" ")))
    return len(w1 & w2)

def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(),row['question1'].split(" "))) # map(function,iterable) - applies the function to each iterable
    w2 = set(map(lambda word: word.lower().strip(),row['question2'].split(" ")))
    return len(w1)+len(w2)



# ADVANCE FEATURE EXTRACTION

# 1. tokenized features
def fetch_token_features(row):

    stop_words = stopwords.words('english')
    safe_div = 1e-8

    q1 = row['question1']
    q2 = row['question2']

    token_features = [0.0]*8

    # converting the sentence into tokens 

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # get non-stop words
    q1_words = set([word for word in q1_tokens if word not in stop_words])
    q2_words = set([word for word in q2_tokens if word not in stop_words])

    # get stopwords
    q1_stop = set([word for word in q1_tokens if word in stop_words])
    q2_stop = set([word for word in q2_tokens if word in stop_words])

    # get common non-stopwords from question pair
    common_word_count = len(q1_words & q2_words)

    # get common stopwords from question pair
    common_stop_count = len(q1_stop & q2_stop)

    # get common tokens from question pair
    common_token_count = len(set(q1_tokens) & set(q2_tokens))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + safe_div)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + safe_div)
    token_features[2] = common_stop_count / (min(len(q1_stop), len(q2_stop)) + safe_div)
    token_features[3] = common_stop_count / (max(len(q1_stop), len(q2_stop)) + safe_div)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + safe_div)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + safe_div)

    # last word for both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])

    # first word for both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features


# 2. length features
def fetch_length_features(row):

    q1 = row['question1']
    q2 = row['question2']

    length_features = [0.0] * 3

    # converting the sentence into tokens 

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features
    
    # absolute length features 

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))

    # average token length of both questions

    length_features[1] =(len(q1_tokens) + len(q2_tokens))/2

    strs = list(distance.lcsubstrings(q1,q2))
    if len(strs)==0:
        length_features[2] = 0
    else:
        length_features[2] = len(strs[0])/(min(len(q1),len(q2)))

    return length_features


# 3. fuzzy features
def fetch_fuzzy_features(row):

    q1 = row['question1']
    q2 = row['question2']

    fuzzy_features = [0.0] *4

    # fuzzy ratio
    fuzzy_features[0] = fuzz.QRatio(q1,q2)

    # fuzz_partial ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1,q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1,q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1,q2)

    return fuzzy_features




def input_query(q1,q2):

    input_col = [] # will store rest features

    q1 = pre_process(q1)
    q2 = pre_process(q2)

    input_col.append(len(q1))
    input_col.append(len(q2))

    input_col.append(len(q1.split()))
    input_col.append(len(q2.split()))

    input_col.append(common_words({'question1':q1,'question2':q2}))
    input_col.append(total_words({'question1':q1,'question2':q2}))

    input_col.append(round(input_col[4]/(input_col[5] + 1e-8),2))
    
    for i in fetch_token_features({'question1':q1,'question2':q2}):
        input_col.append(i)
    
    for i in fetch_length_features({'question1':q1,'question2':q2}):
        input_col.append(i)
    
    for i in fetch_fuzzy_features({'question1':q1,'question2':q2}):
        input_col.append(i)

    q1_tf = tf.transform([q1])
    q2_tf = tf.transform([q2])

    arr = scalar.transform([input_col])
    arr = csr_matrix(arr)

    return hstack((q1_tf,q2_tf,arr))


# api endpoints

@app.get("/health")
def health():
    return {"status" : "ok"}


@app.post("/predict")
def predict_duplicate(user_input : Question_input):
    
    q1 = user_input.question1
    q2 = user_input.question2
    
    input_arr = input_query(q1, q2)
    
    pred = model.predict(input_arr)
    proba = model.predict_proba(input_arr)[0][1]
    
    if pred[0]:
        result = "Duplicate"
    else:
        result = "Not Duplicate"
    return JSONResponse(status_code=200,content={"prediction" : result,"confidence" : float(proba)})
    
@app.get("/")
def root():
    return {"message": "Quora Duplicate API is live"}
    
    