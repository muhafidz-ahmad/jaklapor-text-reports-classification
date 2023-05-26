import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import *
from keras import backend as K

import streamlit as st
import pickle

import text_preprocessing

# title
st.title("Public Report Classification")

# load model, set cache to prevent reloading
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
    
@st.cache(allow_output_mutation=True)
def load_model():
    dependencies = {'Att': Att()}
    model = tf.keras.models.load_model('models/best_model.h5',
                                       custom_objects=dependencies)
    return model

with st.spinner("Loading Model...."):
    model = load_model()
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
# 10 categories of reports
labels_list=['gangguan ketenteraman dan ketertiban', 'jalan', 'jaringan listrik',
             'parkir liar', 'pelayanan perhubungan', 'pohon',
             'saluran air, kali/sungai', 'sampah', 'tata ruang dan bangunan',
             'transportasi publik']

# get new report
new_report = st.text_input("Apa masalah yang mau dilaporkan?","")

# predict

try:
  st.write("Memprediksi Kategori...")
  with st.spinner("Tunggu sebentar..."):
    preprocess_text = text_preprocessing.preprocess(new_report, stem=True)
    seq_tweet = tokenizer.texts_to_sequences([preprocess_text])
    deskripsi = pad_sequences(seq_tweet, padding='post',
                              maxlen=100, truncating='post')
    prediction = model.predict(deskripsi, verbose=0)
    classes = np.argmax(prediction, axis = 1)
    dict_classes = dict(zip(range(len(labels_list)),
                            labels_list))
    pred_class = dict_classes[classes[0]]
    st.write("Prediksi Kategori Laporan:", pred_class)
except:
  st.write("Ada kesalahan :(")
