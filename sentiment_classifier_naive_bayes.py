
import tensorflow as tf
import string
import nltk
import sqlite3
import traceback
import logging
import time
import random
import datetime
import numpy as np
import pandas as pd
import os
import pickle

stop_word_list = list(nltk.corpus.stopwords.words('english'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBsentimentClassifier():
    def __init__(self):
        self.classifier = None
        self.create_sets()

    def create_sets(self, test_size = .01):
        inputs = get_input()
        random.shuffle(inputs)

        self.common_words = get_most_common_words(inputs, 1000)
        inputs_formatted = transform_all_sentences_to_input(inputs, self.common_words)

        training_inputs = inputs_formatted[0:int((1-test_size)*len(inputs))]
        testing_inputs = inputs_formatted[int((1-test_size)*len(inputs)):]

        self.classifier = nltk.NaiveBayesClassifier.train(training_inputs)
        logger.info(nltk.classify.accuracy(self.classifier , testing_inputs))
        self.save_classifier(self.classifier)

    def predict(self, input):
        features = transform_sentence_to_input((input, None), self.common_words)
        if self.classifier.classify(features[0]) == 1:
            return [1, 0]
        else:
            return [0, 1]

    def save_classifier(self, classifier):
       f = open('classifier.pickle', 'wb')
       pickle.dump(classifier, f, -1)
       f.close()

    def load_classifier(self):
       f = open('classifier.pickle', 'rb')
       self.classifier = pickle.load(f)
       f.close()



def transform_all_sentences_to_input(inputs, common_words):
    t_inputs = []
    for i in inputs:
        t_inputs.append(transform_sentence_to_input(i, common_words))
    return t_inputs

def transform_sentence_to_input(input, common_words ):
    result = {}
    for i in common_words:
        result[i] = False
        for j in clean_and_tokenize(input[0]):
            result[i] = True if i == j else False
    return (result, input[1])

def get_most_common_words(input_list, num):
    full_text = []
    for i in input_list:
        full_text.extend(clean_and_tokenize(i[0]))
    freqdist = nltk.FreqDist(full_text)
    return [i[0] for i in freqdist.most_common(num)]

def clean_and_tokenize(input_text):
    clean_text = remove_punctuation_from_text(input_text.lower())
    return remove_stopwords_from_list(nltk.tokenize.word_tokenize(clean_text))

def remove_stopwords_from_list(input_list):
    results = []
    for i in input_list:
        if i not in stop_word_list:
            results.append(i)
    return results

def remove_punctuation_from_text(input_text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in input_text if ch not in exclude)

def get_input():
    inputs = []
    df = pd.read_csv('SAD.csv', error_bad_lines=False)
    for index, row in df.iterrows():
        inputs.append([row[3], row[1]])
    return inputs

if __name__ == '__main__':
    sentiment_classifier = NBsentimentClassifier()
