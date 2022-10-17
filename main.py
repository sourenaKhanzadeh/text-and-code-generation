#import tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow import keras

class TextParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text = self.read_text()
        self.vocab = self.create_vocab()
        self.vocab_size = len(self.vocab)
        self.char2idx = {u: i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)
        self.text_as_int = np.array([self.char2idx[c] for c in self.text])

    def read_text(self):
        text = open(self.file_path, 'rb').read().decode(encoding='utf-8')
        return text

    def create_vocab(self):
        vocab = sorted(set(self.text))
        return vocab

class Model:
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts([self.text])
        self.total_words = len(self.tokenizer.word_index) + 1
        self.input_sequences = []
        self.create_input_sequences()
        self.model = self.create_model()

    def create_input_sequences(self):
        for line in self.text.split(''):
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                self.input_sequences.append(n_gram_sequence)

    def create_model(self):
        model = Sequential()
        model.add(Embedding(self.total_words, 100, input_length=self.seq_length))
        model.add(LSTM(150, return_sequences=True))
        model.add(LSTM(100))
        model.add(Dense(self.total_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, epochs):
        input_sequences = np.array(pad_sequences(self.input_sequences, maxlen=self.seq_length, padding='pre'))
        xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
        ys = to_categorical(labels, num_classes=self.total_words)
        history = self.model.fit(xs, ys, epochs=epochs, verbose=1)
        return history

def to_categorical(labels, num_classes=None):
    return tf.keras.utils.to_categorical(labels, num_classes)

def clean_text(doc):
    tokens = doc.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens


def create_lines(text, seq_length):
    lines = []
    for i in range(seq_length, len(text)):
        seq = tokens[i-length:i]
        line = ' '.join(seq)
        lines.append(line)
        if i > 200000:
            break
    return lines


def save_model(model, model_name):
    model.save(model_name)

def load_model(model, model_name):
    model = tf.keras.models.load_model(model_name)
    return model


def generate_text_seq(model, tokenizer, text_seq_length, seed_text, n_words):
    text = []
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen=text_seq_length, truncating='pre')
        predicted_word_ind = np.argmax(model.predict(encoded), axis=-1)
        predicted_word = tokenizer.index_word[predicted_word_ind[0]]
        seed_text = seed_text + ' ' + predicted_word
        text.append(predicted_word)
    return ' '.join(text)


def train():
    text_parser = TextParser('shakespear.txt')
    model = Model(text_parser.text, 10)
    history = model.train(10)
    save_model(model.model, 'model.h5')
    return model


def get_model_accuracy(model, xs, ys):
    loss_accuray = model.evaluate(xs, ys, verbose=0)
    return loss_accuray

if __name__ == '__main__':
    t = TextParser('shakespear.txt')
    data = t.text.split('\n')
    data = data[253:]
    data = ' '.join(data)

    tokens = clean_text(data)

    length = 50 + 1
    lines = create_lines(tokens, length)


    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)

    sequences = np.array(sequences)
    X, y = sequences[:, :-1], sequences[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    vocab_size = len(tokenizer.word_index) + 1

    y = to_categorical(y, num_classes=vocab_size)

    seq_length = X.shape[1]

    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    # model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    # model.fit(X, y, batch_size=256, epochs=100)

    # save_model(model, 'shakespear_model.h5')
    model = load_model(model, 'shakespear_model.h5')

    # print(generate_text_seq(model, tokenizer, seq_length, 'shall i compare thee to a summer', 24))
    # prepare x_test and y_test for evaluation
    X_test = pad_sequences(X_test, maxlen=seq_length, padding='pre')
    y_test = to_categorical(y_test, num_classes=vocab_size)
    print(X_test.shape, y_test.shape)
    print(get_model_accuracy(model, X_test, y_test))
