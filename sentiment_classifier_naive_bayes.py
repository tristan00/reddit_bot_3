
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
import sys

stop_word_list = list(nltk.corpus.stopwords.words('english'))

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class NBsentimentClassifier():
    def __init__(self, data_memoization):
        self.data_memoization = data_memoization
        self.classifier = None
        #self.create_sets()
        try:
            self.load_classifier()
        except:
            traceback.print_exc()
            self.create_sets()

    def create_sets(self, test_size = .01):
        logger.info('starting to build classifier')
        inputs = get_input()
        random.shuffle(inputs)
        logger.info('inputs read,')

        self.common_words = get_most_common_words(inputs, 1000)
        inputs_formatted = self.transform_all_sentences_to_input(inputs, self.common_words)
        logger.info('inputs formatted')

        training_inputs = inputs_formatted[0:int((1-test_size)*len(inputs))]
        testing_inputs = inputs_formatted[int((1-test_size)*len(inputs)):]

        self.classifier = nltk.NaiveBayesClassifier.train(training_inputs)
        logger.info(nltk.classify.accuracy(self.classifier , testing_inputs))
        self.save_classifier()

    def predict(self, input):
        features = self.transform_sentence_to_input((input, None), self.common_words)
        if self.classifier.classify(features[0]) == 1:
            return [1, 0]
        else:
            return [0, 1]

    def save_classifier(self):
        with open('models/sentiment_classifier.pickle', 'wb') as f1:
            pickle.dump(self.classifier, f1, protocol=pickle.HIGHEST_PROTOCOL)

        with open('models/sentiment_classifier_lables.pickle', 'wb') as f2:
            pickle.dump(self.common_words, f2, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info('model saved')

    def load_classifier(self):
        with open('models/sentiment_classifier.pickle', 'rb') as f:
            self.classifier = pickle.load(f)

        with open('models/sentiment_classifier_lables.pickle', 'rb') as f:
            self.common_words = pickle.load(f)

    def transform_all_sentences_to_input(self, inputs, common_words):
        t_inputs = []
        common_word_set = set(common_words)
        for count, i in enumerate(inputs):
            if count % 1000 == 0:
                logger.info('processed: {0}'.format(count))
            t_inputs.append(self.transform_sentence_to_input(i, common_word_set))
        return t_inputs

    def transform_sentence_to_input(self, inputs, common_words_set ):
        result = dict.fromkeys(common_words_set, False)

        for j in self.data_memoization.clean_and_tokenize(inputs[0]):
            if j in common_words_set:
                result[j] =True
        return (result, inputs[1])

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
    logger.info('here')
    sentiment_classifier = NBsentimentClassifier()
