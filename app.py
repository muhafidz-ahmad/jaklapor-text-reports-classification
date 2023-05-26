import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import *
from keras import backend as K

import requests
import streamlit as st

# title
st.title("Public Report Classification")

# load model, set cache to prevent reloading
@st.cache(allow_output_mutation=True)

dependencies = {'Att': Att()}

class Att(Layer):
    def __init__(self, **kwargs):
        super(Att,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(Att, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        return K.sum(context, axis=1)

def load_model():
    model=tf.keras.models.load_model('models/best_model.h5',
                                     custom_objects=dependencies)
    return model

with st.spinner("Loading Model...."):
    model=load_model()
    
# 10 categories of reports
labels_list=['gangguan ketenteraman dan ketertiban', 'jalan', 'jaringan listrik',
             'parkir liar', 'pelayanan perhubungan', 'pohon',
             'saluran air, kali/sungai', 'sampah', 'tata ruang dan bangunan',
             'transportasi publik']

# text preprocessing
alay_df = pd.read_csv('data/new_kamusalay.csv', 
                      encoding = 'latin-1', 
                      header = None)

alay_df.rename(columns={0: 'original', 
                        1: 'replacement'},
               inplace = True)

alay_dict_map = dict(zip(alay_df['original'], alay_df['replacement']))

new_alay = dict()

def normalize_alay(text):
  global new_alay
  for word in text.split():
    if word in alay_dict_map:
      if word not in new_alay:
        new_alay[word] = alay_dict_map[word]
  return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

def convert_lower_case(text):
  return text.lower()

def remove_stop_words(text):
  stop_words = stopwords.words('indonesian')
  words = word_tokenize(str(text))
  new_text = ""
  for w in words:
    if w not in stop_words and len(w) > 1:
      new_text = new_text + " " + w
  return new_text

def remove_unnecessary_char(text):
  text = re.sub('permasalahan:\n',' ',text) # Remove every 'Permasalahan:\n'
  text = re.sub('  +', ' ', text) # Remove extra spaces
  text = re.sub(' jam ', ' ', text) # Remove tulisan "jam"
  text = re.sub(' pagi ', ' ', text) # Remove tulisan "pagi"
  text = re.sub(' sore ', ' ', text) # Remove tulisan "sore"
  i_text = text.find('lokasi:')
  if i_text != -1:
    text = text[:i_text]
  return text

def remove_punctuation(text):
  symbols = string.punctuation
  for i in range(len(symbols)):
    text = text.replace(symbols[i], ' ')
    text = text.replace("  ", " ")
  text = text.replace(',', '')
  return text

def stemming(text):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  
  tokens = word_tokenize(str(text))
  new_text = ""
  for w in tokens:
    new_text = new_text + " " + stemmer.stem(w)
  return new_text

counter = 0

def preprocess(text, stem=False, verbose=0):
  global counter

  text = convert_lower_case(text)
  text = remove_unnecessary_char(text)
  text = remove_punctuation(text)
  text = normalize_alay(text)
  text = remove_stop_words(text)
  if stem == True:
    text = stemming(text)
  text = text.strip()

  counter += 1
  if (counter % 1 == 0) and (verbose == 1):
    print(f"\r{counter}", end="")
  return text

# get new report
new_report = st.text_input("Apa masalah yang mau dilaporkan?","")

# predict

try:
  st.write("Predicting Class...")
  with st.spinner("Classifying..."):
    preprocess_text = preprocess(new_report, verbose=0)
    seq_tweet = tokenizer.texts_to_sequences([preprocess_text])
    deskripsi = pad_sequences(seq_tweet, padding='post',
                              maxlen=100, truncating='post')
    prediction = model.predict(deskripsi, verbose=0)
    classes = np.argmax(prediction, axis = 1)
    dict_classes = dict(zip(range(len(labels_list)),
                            labels_list))
    pred_class = dict_classes[classes[0]]
    st.write("Predicted Class:", pred_class)
except:
  st.write("Ada kesalahan :(")
