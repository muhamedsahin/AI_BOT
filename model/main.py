import json
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Veri setini yükleme
with open('./data/chat.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

kullanici = []
bot = []

for veri in data:
    kullanici.append(veri["soru"])
    bot.append(veri["cevap"])

# Tokenizer oluşturma ve eğitme
tokenizer = Tokenizer()
tokenizer.fit_on_texts(kullanici + bot)

# Kelime dağarcığını alma
word_index = tokenizer.word_index

# Kullanıcı girişlerini ve bot yanıtlarını sayılara dönüştürme
kullanici_sequences = tokenizer.texts_to_sequences(kullanici)
bot_sequences = tokenizer.texts_to_sequences(bot)

# Giriş ve çıkış dizilerini ayarlama
max_sequence_length = 50  # Belirli bir maksimum dizgi uzunluğu seçin
X = pad_sequences(kullanici_sequences, maxlen=max_sequence_length, padding='post')
y = pad_sequences(bot_sequences, maxlen=max_sequence_length, padding='post')

# Modeli oluşturma
vocab_size = len(word_index) + 1
embedding_dim = 100
hidden_units = 256

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
model.add(LSTM(hidden_units, return_sequences=True))
model.add(Dense(vocab_size, activation='softmax'))

# Modeli derleme
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğitme
epochs = 100
batch_size = 32
model.fit(X, y, epochs=epochs, batch_size=batch_size)

# Modeli kaydetme
model.save("chat.h5")

# Tokenizer'ı kaydetme
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Modelin değerlendirilmesi
test_loss, test_accuracy = model.evaluate(X, y, verbose=2)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
