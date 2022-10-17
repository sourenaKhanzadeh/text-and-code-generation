from lib2to3.pgen2.tokenize import tokenize, untokenize
import tensorflow  as tf
import numpy  as np
import matplotlib.pyplot  as plt
import os
import string
import io
from tensorflow.keras.preprocessing.text  import Tokenizer
from tensorflow.keras.utils  import to_categorical
from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers  import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence  import pad_sequences
from sklearn.model_selection  import train_test_split
from tokenize import tokenize, untokenize
import keyword

class CodeParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text = self.read_text()
        self.vocab = self.create_vocab()
        self.vocab_size = len(self.vocab)
        self.char2idx = {u: i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)
        self.text_as_int = np.array([self.char2idx[c] for c in self.text])

    def read_text(self):
        text = open(self.file_path, 'r', encoding='utf-8').readlines()
        return text

    def create_vocab(self):
        vocab = sorted(set(self.text))
        return vocab
    
    def make_question_answer(self):
        dps = []
        dp = None
        for line in self.text:
            if line[0] == "#":
                if dp:
                    dp['solution'] = ''.join(dp['solution'])
                    dps.append(dp)
                dp = {"question": None, "solution": []}
                dp['question'] = line[1:]
            else:
                dp["solution"].append(line)
        return dps
    
    def tokenize_python_code(self, string):
        python_tokens = list(tokenize(io.BytesIO(string.encode('utf-8')).readline))
        tokenized_output = []
        for i in range(0, len(python_tokens)):
            tokenized_output.append((python_tokens[i].type, python_tokens[i].string))
        return tokenized_output



class Model:
    def __init__(self, text):
        self.text = text
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts([self.text])
        self.total_words = len(self.tokenizer.word_index) + 1
        self.input_sequences = self.tokenizer.texts_to_sequences([self.text])
        self.predictors, self.label, self.max_sequence_len = self.preprocessing()
        self.seq_length = self.predictors.shape[1]
        self.create_input_sequences()
        self.model = self.create_model()

    def preprocessing(self):
        max_sequence_len = max([len(x) for x in self.input_sequences])
        input_sequences = np.array(pad_sequences(self.input_sequences, maxlen=max_sequence_len, padding='pre'))
        predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
        label = to_categorical(label, num_classes=self.total_words)
        return predictors, label, max_sequence_len

    def create_input_sequences(self):
        for line in self.text.split(' '):
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
        history = self.model.fit(self.predictors, self.label, epochs=epochs, verbose=1)
        return history

    def generate_text(self, seed_text, n_words):
        text = []
        for _ in range(n_words):
            encoded = self.tokenizer.texts_to_sequences([seed_text])[0]
            encoded = pad_sequences([encoded], maxlen=self.seq_length, truncating='pre')
            predicted_word_ind = np.argmax(self.model.predict(encoded), axis=-1)
            predicted_word = self.tokenizer.index_word[predicted_word_ind[0]]
            seed_text = seed_text + ' ' + predicted_word
            text.append(predicted_word)
        return ' '.join(text)
    
    def save_model(self, model_name):
        self.model.save(model_name)
    
    def load_model(self, model_name):
        self.model = tf.keras.models.load_model(model_name)

def train_model():
    parser = CodeParser('english_python_data.txt')
    # tokenizer = Tokenizer()
    model = Model(parser.text)
    # history = model.train(5)
    # model.save_model('python_generator.h5')
    model.load_model('python_generator.h5')
    # calculate the max sequence length
    max_sequence_len = model.max_sequence_len
    print(model.generate_text("get the maximum of two list", 10))


def main():
    parser =  CodeParser('english_python_data.txt')
    dps = parser.make_question_answer()
    print(parser.tokenize_python_code(dps[1]['solution']))
    print(untokenize(parser.tokenize_python_code(dps[1]['solution'])).decode('utf-8'))


if __name__ == '__main__':
    main()