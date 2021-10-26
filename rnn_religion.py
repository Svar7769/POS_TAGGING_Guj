import nltk
from nltk.tag import hmm
from nltk.stem import PorterStemmer
from nltk.corpus.reader import TaggedCorpusReader
from nltk.corpus.reader import PlaintextCorpusReader
from sklearn import metrics


tagged_corpus = TaggedCorpusReader(".","Data/guj_religion_set1.txt")
tagged_sentences = tagged_corpus.tagged_sents()
tagged_word = tagged_corpus.tagged_words()

# tagged_sentences = tagged_corpus.tagged_sents()
# tagged_words = tagged_corpus.tagged_words()

# print (tagged_sentences[4])
# print (tagged_words[0])

print("No. of tagged sentences for training: " , len(tagged_sentences))
print("No. of tagged words for training: " , len(tagged_word))

import numpy as np

sentences, sentence_tags = [], []
for tagged_sentence in tagged_sentences:
    sentence, tags = zip(*tagged_sentence)
    sentences.append(np.array(sentence))
    sentence_tags.append(np.array(tags))

print(sentences[5])
print(sentence_tags[5])

from sklearn.model_selection import train_test_split

(train_sentences,
 test_sentences,
 train_tags,
 test_tags) = train_test_split(sentences, sentence_tags, test_size=0.2)

words, tags = set([]), set([])

for s in train_sentences:
    for w in s:
        words.add(w.lower())

for ts in train_tags:
    for t in ts:
        tags.add(t)

word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['-PAD-'] = 0  # The special value used for padding
word2index['-OOV-'] = 1  # The special value used for OOVs

tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
tag2index['-PAD-'] = 0  # The special value used to padding

train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

for s in train_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    train_sentences_X.append(s_int)

for s in test_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    test_sentences_X.append(s_int)

for s in train_tags:
    train_tags_y.append([tag2index[t] for t in s])

for s in test_tags:
    test_tags_y.append([tag2index[t] for t in s])

# print(train_sentences_X[0])
# print(test_sentences_X[0])
# print(train_tags_y[0])
# print(test_tags_y[0])


MAX_LENGTH = len(max(train_sentences_X, key=len))
print(MAX_LENGTH)

from keras.preprocessing.sequence import pad_sequences

train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')

# print(train_sentences_X[0])
# print(test_sentences_X[0])
# print(train_tags_y[0])
# print(test_tags_y[0])

from tensorflow import keras

from keras import backend as K

def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)

        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy

    return ignore_accuracy

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation, SimpleRNN

model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH,)))
model.add(Embedding(len(word2index), 128))
model.add(SimpleRNN(256, return_sequences=True))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(0.001),
              metrics=['accuracy',f1_m,precision_m, recall_m])

model.summary()

def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
# print(cat_train_tags_y[0])

print(train_tags_y)

model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=15, validation_split=0.2)

loss, accuracy, f1_score, precision, recall = model.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))
print("Accuracy: " + str(accuracy))
print("f1_score: " + str(f1_score))
print("precision: " + str(precision))
print("recall: " + str(recall))
