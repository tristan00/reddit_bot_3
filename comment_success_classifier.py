#this model will be a simple dnn which extract features from
# bags of words and timestamps to classify comments by success
#no model savings as

import datetime
import logging
import random
import sqlite3
import time
import string
import nltk
import numpy as np
import tensorflow as tf
import pickle
import operator
import traceback
import os
import re

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#TODO: store ngrams in db to allow model storing

nodes_per_layer = 3000
max_results_to_analyze = 10000000
get_newest_results = True #if i cut out some results, this will only get newer results keeping my bot more updated in the meta
stop_word_list = list(nltk.corpus.stopwords.words('english'))
stop_word_set = set(stop_word_list)

num_of_score_buckets = 10
num_of_features_per_n = 100
min_n_gram = 2
max_n_gram = 4
max_topic = 3
subreddit_list = []
model_save_location = 'models/comment_success_classifier_model_10L_final.ckpt'




n_classes = 10

class DNN_comment_classifier():
    def __init__(self, num_of_topics, topic_minimum_probability_threshold, retrain = False):
        self.input_width = 47+900+300+6+144 #better to keep just undr a power of 2
        self.sentiment_memoization = {}
        self.topic_memoization = {}
        self.n_gram_dicts = {}
        self.num_of_topics = num_of_topics
        self.topic_minimum_probability_threshold = topic_minimum_probability_threshold
        self.retrain = retrain
        self.sub_list = get_subreddit_list()
        self.optimizer, self.cost, self.x, self.y, self.sess, self.prediction, self.saver, self.prob = self.build_neural_network()

        if retrain:
            self.read_metadata(num_of_features_per_n)
        else:
            self.load_metadata()
            self.load_model(self.saver, self.sess)
        #self.input_width = self.get_input_size()

    def run_input(self, i):
        model_results = self.sess.run(self.prediction, feed_dict = {self.x:[i]}).tolist()[0]
        return sum(map(operator.mul, model_results, self.bucket_avg))

    def train_nn(self, epochs, sentiment_classifier, topic_model):
        self.train_neural_network(epochs, self.optimizer, self.cost, self.x, self.y, self.sess, self.prediction, sentiment_classifier, topic_model)

    def build_neural_network(self):
        x = tf.placeholder('float', [None, self.input_width])
        y = tf.placeholder('float', [None, n_classes])
        prediction, prob = self.neural_network_model(nodes_per_layer, x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        saver = tf.train.Saver(tf.global_variables())
        sess = tf.Session()
        if self.retrain:
            sess.run(tf.global_variables_initializer())
        else:
            self.load_model(saver, sess)
        return optimizer, cost, x, y, sess, prediction, saver, prob

    def neural_network_model(self, nodes_per_layer, x):
        hidden_1_layer = {'weights': tf.Variable(tf.random_normal([self.input_width, nodes_per_layer])),
                          'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
        hidden_2_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                          'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
        hidden_3_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                          'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
        hidden_4_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                          'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
        hidden_5_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                          'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
        hidden_6_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                          'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
        hidden_7_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                          'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
        hidden_8_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                          'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
        hidden_9_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                          'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
        hidden_10_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                          'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
        output_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, n_classes])),
                            'biases': tf.Variable(tf.random_normal([n_classes]))}

        keep_prob = .5
        prob = tf.placeholder_with_default(1.0, shape=())
        l1 = tf.add(tf.matmul(x, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)
        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)
        l2_dropout = tf.nn.dropout(l2, prob)
        l3 = tf.add(tf.matmul(l2_dropout, hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.relu(l3)
        l3_dropout = tf.nn.dropout(l3, prob)
        l4 = tf.add(tf.matmul(l3_dropout, hidden_4_layer['weights']), hidden_4_layer['biases'])
        l4 = tf.nn.relu(l4)
        l4_dropout = tf.nn.dropout(l4, prob)
        l5 = tf.add(tf.matmul(l4_dropout, hidden_5_layer['weights']), hidden_5_layer['biases'])
        l5 = tf.nn.relu(l5)
        l5_dropout = tf.nn.dropout(l5, prob)
        l6 = tf.add(tf.matmul(l5_dropout, hidden_6_layer['weights']), hidden_6_layer['biases'])
        l6 = tf.nn.relu(l6)
        l6_dropout = tf.nn.dropout(l6, prob)
        l7 = tf.add(tf.matmul(l6_dropout, hidden_7_layer['weights']), hidden_7_layer['biases'])
        l7 = tf.nn.relu(l7)
        l7_dropout = tf.nn.dropout(l7, prob)
        l8 = tf.add(tf.matmul(l7_dropout, hidden_8_layer['weights']), hidden_8_layer['biases'])
        l8 = tf.nn.relu(l8)
        l8_dropout = tf.nn.dropout(l8, prob)
        l9 = tf.add(tf.matmul(l8_dropout, hidden_9_layer['weights']), hidden_9_layer['biases'])
        l9 = tf.nn.relu(l9)
        l9_dropout = tf.nn.dropout(l9, prob)
        l10 = tf.add(tf.matmul(l9_dropout, hidden_10_layer['weights']), hidden_10_layer['biases'])
        l10 = tf.nn.relu(l10)
        l10_dropout = tf.nn.dropout(l10, prob)
        output = tf.matmul(l10_dropout, output_layer['weights']) +  output_layer['biases']
        return output, prob

    def train_neural_network(self, epochs, optimizer, cost, x, y, sess, prediction, sentiment_classifier, topic_model):
        start_time = time.time()
        batch_size = 100
        hm_epochs = epochs
        inputs = get_db_input()
        logger.info('got inputs')
        random.shuffle(inputs)
        train_x, train_y, test_x, test_y = self.create_feature_sets_and_labels(inputs,  sentiment_classifier, topic_model)
        del inputs[:]
        logger.info('input len: {0}'.format(len(train_x)))

        logger.info('starting training')
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict= {x:batch_x, y:batch_y, self.prob: 0.5})
                epoch_loss += c
                i += batch_size
                logger.info("Batch {0} of epoch {1} completed, loss: {2}, time:{3}".format(i/batch_size, epoch, c, time.time() - start_time))
            logger.info("Epoch {0} completed out of {1}, loss: {2}".format(epoch, hm_epochs,epoch_loss))
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accuracy_float = accuracy.eval(session = sess, feed_dict = {x:test_x, y:test_y, self.prob: 1.0})
        logger.info(('Accuracy:', accuracy_float))
        self.save_model()
        return sess, prediction, x, y

    def create_feature_sets_and_labels(self, inputs,  sentiment_classifier, topic_model, test_size = .01):
        random.shuffle(inputs)
        feature_list = []
        for count, i in enumerate(inputs):
            if count%1000 == 0:
                logger.info('comment classifier proccessed {0} comments, timestamp:{1}'.format(count, time.time()))
            feature_list.append([self.create_input_features(i, sentiment_classifier, topic_model), self.create_output_features(i)])
        testing_size = int(test_size*len(inputs))
        train_x = [i[0] for i in feature_list[testing_size:]]
        train_y = [i[1] for i in feature_list[testing_size:]]
        test_x = [i[0] for i in feature_list[:testing_size]]
        test_y = [i[1] for i in feature_list[:testing_size]]
        return train_x, train_y, test_x, test_y

    def create_output_features(self, output):
        output_array = np.zeros(num_of_score_buckets) #[0 for i in range(num_of_score_buckets)]
        for i in range(len(self.border_values_for_upvotes)):
            if output[6] < self.border_values_for_upvotes[i]:
                output_array[i] = 1
                return output_array
        output_array[len(self.border_values_for_upvotes)] = 1
        return output_array

    def create_input_feature_from_text(self, title, parent_text, child_text, title_time_stamp, parent_time_stamp, child_time_stamp, subreddit, sentiment_classifier, topic_model):
        tokenized_title = clean_and_tokenize(title)
        tokenized_parent = clean_and_tokenize(parent_text)
        tokenized_child = clean_and_tokenize(child_text)

        parent_sentiment = self.get_sentiment_classification(tokenized_parent, sentiment_classifier, tokenized=True)
        child_sentiment = self.get_sentiment_classification(tokenized_child, sentiment_classifier, tokenized=True)
        post_sentiment = self.get_sentiment_classification(tokenized_title, sentiment_classifier, tokenized=True)
        parent_topic = self.get_topic_classification(tokenized_parent, topic_model, self.num_of_topics, tokenized=True)
        child_topic = self.get_topic_classification(tokenized_child, topic_model, self.num_of_topics, tokenized=True)
        post_topic = self.get_topic_classification(tokenized_title, topic_model, self.num_of_topics, tokenized=True)
        parent_timestamp_features = create_timestamp_features(parent_time_stamp)
        child_timestamp_features = create_timestamp_features(child_time_stamp)
        post_timestamp_features = create_timestamp_features(title_time_stamp)
        subreddit_features = get_subreddit_features(subreddit, self.sub_list)
        parent_features = get_text_features(parent_text, self.n_gram_orders_dict, tokenized=True)
        child_features = get_text_features(child_text, self.n_gram_orders_dict, tokenized=True)
        title_features = get_text_features(title, self.n_gram_orders_dict, tokenized=True)
        input_features = np.concatenate((parent_sentiment, child_sentiment, post_sentiment, parent_topic, child_topic, post_topic,
                                         parent_timestamp_features, child_timestamp_features, post_timestamp_features,
                                        subreddit_features, parent_features, child_features,title_features))
        return input_features

    def create_input_features(self, i, sentiment_classifier, topic_model):
        return self.create_input_feature_from_text( i[21], i[14], i[5], i[24], i[7], i[16], i[2], sentiment_classifier, topic_model)

    #build ngrams and rank comments
    #for each comment, remove stopwords then place into n-grams
    #sort the ngrams by how common they are, store the most common ones
    def read_metadata(self, num_of_features_per_n):
        score_list = []
        self.n_gram_orders_dict = {}

        for n in range(min_n_gram, max_n_gram+1):
            self.n_gram_dicts.setdefault(n, {})
        res = get_db_input()
        comments = []
        for r in res:
            score_list.append(r[15])
            comments.append(clean_and_tokenize(r[21]))
            comments.append(clean_and_tokenize(r[14]))
            comments.append(clean_and_tokenize(r[5]))
        for c in comments:
            for n in self.n_gram_dicts.keys():
                if len(c) >= n:
                    for i in range(len(c) - n):
                        current_value = self.n_gram_dicts[n].get(' '.join(c[i:i+n]), 0)
                        self.n_gram_dicts[n][' '.join(c[i:i+n])] = current_value + 1
                else:
                    break
        for n in self.n_gram_dicts.keys():
            self.n_gram_orders_dict[n] = get_dict_keys_sorted_by_values(self.n_gram_dicts[n], num_of_features_per_n)
        self.border_values_for_upvotes = get_border_values(num_of_score_buckets, score_list)#gets the border of the buckets for the output features
        self.bucket_avg = create_average_of_buckets(self.border_values_for_upvotes, score_list)
        self.save_metadata()

    def save_model(self):
        self.saver.save(self.sess, model_save_location)
        logger.info('model saved')

    def load_model(self, saver, sess):
        saver.restore(sess, model_save_location)

    def save_metadata(self):
        with open('models/comment_classifier_ngrams.pickle', 'wb') as f1:
            pickle.dump(self.n_gram_orders_dict, f1, protocol=pickle.HIGHEST_PROTOCOL)

        with open('models/comment_classifier_upvote_border_values.pickle', 'wb') as f2:
            pickle.dump(self.border_values_for_upvotes, f2, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info('metadata saved')

    def load_metadata(self):
        with open('models/comment_classifier_ngrams.pickle', 'rb') as f1:
            self.n_gram_orders_dict = pickle.load(f1)

        with open('models/comment_classifier_upvote_border_values.pickle', 'rb') as f2:
            self.border_values_for_upvotes = pickle.load(f2)

    def get_sentiment_classification(self, text, sentiment_classifier, tokenized = False):
        return self.sentiment_memoization.setdefault((tuple(text), tokenized),
                                                     np.asarray(sentiment_classifier.predict(tuple(text), tokenized = tokenized)))

    def get_topic_classification(self, text, topic_model, num_of_topics, tokenized = False):
        results = self.topic_memoization.setdefault((tuple(text), tokenized), topic_model.get_topic(tuple(text), tokenized=tokenized))
        result_vector = [0 for i in range(num_of_topics)]
        results_sorted = sorted(results, key=lambda x: x[1], reverse = True)
        for r in results_sorted[:max_topic]:
            result_vector[r[0]] = 1
        result_vector = np.asarray(result_vector)
        return result_vector

#Feature creation methods:
#features of hour will be most useful followed by weekday and month, week of month is there for now
def create_timestamp_features(timestamp):
    datetime_timestamp = datetime.datetime.utcfromtimestamp(float(timestamp))
    hour_feature = [0 for i in range(24)]
    week_day_feature = [0 for i in range(7)]
    week_month_feature = [0 for i in range(5)]
    month_feature = [0 for i in range(12)]
    hour_feature[datetime_timestamp.hour] = 1
    week_day_feature[datetime_timestamp.weekday()] = 1
    week_month_feature[datetime_timestamp.day//7] = 1
    month_feature[datetime_timestamp.month-1] = 1 #month starts at 1
    np_array =  np.asarray(hour_feature + week_day_feature + week_month_feature + month_feature)
    return np_array

def get_text_features(text, n_gram_dict, tokenized = False):
    word_features = [0 for i in range(len(n_gram_dict.keys())*len(n_gram_dict[list(n_gram_dict.keys())[0]]))]
    index = 0
    if tokenized:
        formatted_word = ' '.join(text)
    else:
        formatted_word = format_text(text)
    for n in n_gram_dict.keys():
        for i in n_gram_dict[n]:
            if i in formatted_word:
                word_features[index] = 1
            index+= 1
    return np.asarray(word_features)

def get_subreddit_features(subreddit, subreddit_list):
    subreddit_features = np.zeros(len(subreddit_list)) #[0 for i in range(len(subreddit_list))]
    subreddit_features[subreddit_list.index(subreddit)] = 1
    return subreddit_features

#Helper methods:
def format_text(input_text):
    return ' '.join(clean_and_tokenize(input_text))

def clean_and_tokenize(input_text):
    clean_text = remove_stopwords_from_comment(remove_punctuation_from_text(input_text.lower()))
    return nltk.tokenize.word_tokenize(clean_text)

def remove_stopwords_from_list(input_list):
    results = []
    for i in input_list:
        if i not in stop_word_set:
            results.append(i)
    return results

def remove_stopwords_from_comment(input_str):
    return  ' '.join([word for word in input_str.split() if word not in stop_word_set])

def remove_punctuation_from_text(input_text):
    return input_text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords_from_list(input_list):
    results = []
    for i in input_list:
        if i not in stop_word_set:
            results.append(i)
    return results

def get_subreddit_features(subreddit, subreddit_list):
    subreddit_features = np.zeros(len(subreddit_list)) #[0 for i in range(len(subreddit_list))]
    subreddit_features[subreddit_list.index(subreddit)] = 1
    return subreddit_features

#Helper methods:
def get_dict_keys_sorted_by_values(d, number_to_return, reverse = True):
    sorting_list = []
    for i in d.items():
        sorting_list.append(i)
    sorting_list = sorted(sorting_list, key=lambda x: x[1], reverse = reverse)
    return [i[0] for i in sorting_list][0:number_to_return]

#precalculate border values so that it does not need to store the scores in memory
#returns list of num_of_buckets - 1 len with the values representing border between buckets
def get_border_values(num_of_buckets, score_list):
    border_values = []
    for i in range(1, num_of_buckets):
        border_values.append(np.percentile(score_list, 100*i/num_of_buckets))
    return border_values

def get_score_bucket_average(num_of_buckets, score_list):
    border_values = []
    for i in range(1, num_of_buckets):
        border_values.append(np.percentile(score_list, 100*i/num_of_buckets))
    return border_values

def create_average_of_buckets(border_list, score_list):
    output_array = [[] for i in range(len(border_list) + 1)] #[0 for i in range(num_of_score_buckets)]
    for score in score_list:
        num_added = False
        for i in range(len(border_list)):
            if score < border_list[i]:
                output_array[i].append(score)
                num_added = True
                break
        if not num_added:
            output_array[-1].append(score)
    average_values_list = []
    for i in output_array:
        try:
            average_values_list.append(sum(i)/len(i))
        except:
            average_values_list.append(0)
    print([len(i) for i in output_array])
    print([sum(i)/(len(i) +.001) for i in output_array])
    return average_values_list

#get parent, child, post data from db
#allows user to only allow data from certai subreddits by passing list of elegible subreddit ids into it
def get_db_input():
    with sqlite3.connect('reddit.db') as conn:
        res = conn.execute('''select *
    from comments a join comments b on a.c_id = b.parent_id
    join posts c on a.p_id = c.p_id order by b.submitted_timestamp desc''').fetchall()
        logger.info('len of input = {0}'.format(len(res)))
        if get_newest_results or max_results_to_analyze > len(res):
            output =  res[:max_results_to_analyze]
        else:
            random.shuffle(res)
            output =  res[:max_results_to_analyze]
        return output

def get_subreddit_list():
    global subreddit_list
    if len(subreddit_list) == 0:
        with sqlite3.connect('reddit.db') as conn:
            res = conn.execute('select distinct s_id from subreddits order by full_name').fetchall()
            subreddit_list = [i[0] for i in res]
    return subreddit_list

#testing
if __name__ == '__main__':
    dnn = DNN_comment_classifier()
    print('here')


