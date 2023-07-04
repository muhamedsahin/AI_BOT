import tensorflow as tf
import pickle
import re
import string
import nltk
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Modeli yükleme
model = tf.keras.models.load_model('chat.h5')

# Tokenizer nesnesini yükleme
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

#Yazıyı uygun hale getirme
def preprocess_text(text):
    # Küçük harfe dönüştürme
    text = text.lower()
    
    # Noktalama işaretlerini kaldırma
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Rakamları kaldırma
    text = re.sub(r'\d+', '', text)
    
    # Tokenizasyon (kelimeleri ayırma)
    tokens = word_tokenize(text)
    
    # Stop kelimelerini kaldırma
    stop_words = set(stopwords.words('turkish'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Kök bulma (stemming)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Önişleme sonucunu birleştirme
    preprocessed_text = ' '.join(stemmed_tokens)
    
    return preprocessed_text

# Tahmin yapmak için kullanılan fonksiyon
def generate_text(input_text):
    #Metin ön işleme
    input_text = preprocess_text(input_text)
    # Metni dönüştürme ve diziye dönüştürme
    sequence = tokenizer.texts_to_sequences([input_text])
    sequence = pad_sequences(sequence, maxlen=50, padding='post')

    # Tahmin yapma
    predicted_sequence = model.predict(sequence)
    predicted_index = tf.argmax(predicted_sequence, axis=-1).numpy()[0]
    predicted_words = []
    for text in predicted_index:
        if text != 0:
            predicted_words.append(tokenizer.index_word[text])

    predicted_sentence = ' '.join(predicted_words)
    return predicted_sentence

while True:

    messages = input("Bota Yaz:")
    response = generate_text(messages)
    print(response)
    print("-------------------------")
    if(messages == "exit"):
        break