import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import json

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

file_path = "./data/data.txt"  # Okunacak dosyanın yolunu belirtin

with open(file_path, "r") as file:
    content = file.read()

#kullanıcı = []
#bot = []
dialog = []

for row in content.split('\n'):
    
    #kullanıcı.append(preprocess_text(row.split('\t')[0]))
    #bot.append(preprocess_text(row.split('\t')[1]))
    data = {
        "soru": preprocess_text(row.split('\t')[0]),
        "cevap": preprocess_text(row.split('\t')[1]),
    }
    
    dialog.append(data)

file_path = "./data/chat.json"

with open(file_path, 'w') as file:
    json.dump(dialog, file)