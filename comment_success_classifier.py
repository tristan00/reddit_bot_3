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
import pandas as pd
from topic_model import Reddit_LDA_Model

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#TODO: store ngrams in db to allow model storing

nodes_per_layer = 3000
max_results_to_analyze = 10000000
stop_word_list = list(nltk.corpus.stopwords.words('english'))
stop_word_set = set(stop_word_list)
max_word_length_for_features = 10
max_word_count_for_features = 10
word_count_bucket_size = 4
sub_feature_dict = {}

num_of_score_buckets = 4
num_of_features_per_n = 100
min_n_gram = 2
max_n_gram = 4
max_topic = 3
subreddit_list = []
model_save_location = 'models/comment_success_classifier_model_10L_final.ckpt'
pos_tag_map = {'CC':1,'CD':2,'DT':3,'EX':4,'FW':5,'IN':6, 'JJ':7,'JJR':8,'JJS':9, 'LS':10,'MD':11,'NN':12, 'NNS':13, 'NNP':14, 'NNPS': 15,
               'PDT':16, 'POS':17, 'PRP':18, 'PRP$':19,'RB':20,'RBR':21, 'RBS':22, 'RP':23, 'SYM':24, 'TO':25,'UH':26, 'VB':27, 'VBD':28, 'VBG':29, 'VBN':30,
               'VBP':31, 'VBZ':32, 'WDT':33, 'WP':34, 'WP$':35, 'WRB':36}

n_classes = num_of_score_buckets

class DNN_comment_classifier():
    def __init__(self, topics, retrain = False):
        self.topics = topics
        self.input_width = 1297+66 + (37*3)
        self.sentiment_memoization = {}
        self.topic_memoization = {}
        self.pos_list = []
        self.retrain = retrain
        self.sub_list = get_subreddit_list()
        self.optimizer, self.cost, self.x, self.y, self.sess, self.prediction, self.saver, self.prob = self.build_neural_network()
        self.read_metadata()


    def run_input(self, i):
        model_results = self.sess.run(self.prediction, feed_dict = {self.x:[i], self.prob: 1})[0]
        return sum(map(operator.mul, model_results, self.bucket_avg))

    def train_nn(self, epochs):
        self.train_neural_network(epochs, self.optimizer, self.cost, self.x, self.y, self.sess, self.prediction)

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

        output_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, n_classes])),
                            'biases': tf.Variable(tf.random_normal([n_classes]))}

        prob = tf.placeholder_with_default(1.0, shape=())
        l1 = tf.add(tf.matmul(x, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.leaky_relu(l1)
        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.leaky_relu(l2)
        l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.leaky_relu(l3)
        l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
        l4 = tf.nn.leaky_relu(l4)
        l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases'])
        l5 = tf.nn.leaky_relu(l5)
        l6 = tf.add(tf.matmul(l5, hidden_5_layer['weights']), hidden_5_layer['biases'])
        l6 = tf.nn.leaky_relu(l6)
        l6_dropout = tf.nn.dropout(l6, prob)
        output = tf.add(tf.matmul(l6_dropout , output_layer['weights']), output_layer['biases'])
        return output, prob

    def train_neural_network(self, epochs, optimizer, cost, x, y, sess, prediction, preprocessed = True):
        start_time = time.time()
        batch_size = 10000
        hm_epochs = epochs
        #inputs = get_db_input_to_pd()
        inputs = get_db_input()
        train_x, train_y, test_x, test_y = self.create_feature_sets_and_labels(inputs, preprocessed = preprocessed)

        logger.info('got inputs')
        del inputs[:]
        logger.info('input len: {0}'.format(len(train_x)))
        logger.info('starting training')
        learning_log = []
        output_log = []


        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accuracy_float = accuracy.eval(session = sess, feed_dict = {x:test_x, y:test_y, self.prob:1})
        learning_log.append((0, accuracy_float, None))
        output_log.append([0, self.run_input(test_x[0])])

        for epoch in range(hm_epochs):
            print('learning log:')
            for i in learning_log:
                print(i)
            print('output_log:')
            for i in output_log:
                print(i)
            epoch_loss = 0
            i=0
            while i < len(train_x):
                try:
                    start = i
                    end = i + batch_size
                    batch_x = np.array(train_x[start:end])
                    batch_y = np.array(train_y[start:end])
                    _, c = sess.run([optimizer, cost], feed_dict= {x:batch_x, y:batch_y, self.prob:.5})
                    epoch_loss += c
                    i += batch_size
                    logger.info("Batch {0} of epoch {1} completed, loss: {2}, time:{3}".format(i/batch_size, epoch+1, c, time.time() - start_time))
                except:
                    traceback.print_exc()
                    list_batch = batch_x.tolist()
                    print(list_batch)

            logger.info("Epoch {0} completed out of {1}, loss: {2}".format(epoch+1, hm_epochs,epoch_loss))
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            #accuracy_float = accuracy.eval(session = sess, feed_dict = {x:test_x, y:test_y, self.prob: 1.0})
            accuracy_float = accuracy.eval(session = sess, feed_dict = {x:test_x, y:test_y, self.prob:1})
            learning_log.append((epoch+1, accuracy_float, epoch_loss))
            output_log.append([epoch+1, self.run_input(test_x[0])])

        print('learning log:')
        for i in learning_log:
            print(i)
        print('output_log:')
        for i in output_log:
            print(i)


        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        #accuracy_float = accuracy.eval(session = sess, feed_dict = {x:test_x, y:test_y, self.prob: 1.0})
        accuracy_float = accuracy.eval(session = sess, feed_dict = {x:test_x, y:test_y, self.prob:1})
        logger.info(('Accuracy:', accuracy_float))
        self.save_model()
        return sess, prediction, x, y

    def validate(self, input_feature, output_features):
        if input_feature.size == self.input_width:
            return True
        else:
            return None

    def create_feature_sets_and_labels(self, inputs, test_size = .01, preprocessed = False):
        global sub_feature_dict
        feature_list = []

        invalid_inputs = 0
        for count, i in enumerate(inputs):
            if count%1000 == 0:
                logger.info('comment classifier proccessed {0} comments, invalid inputs:{1}, timestamp:{2}'.format(count,invalid_inputs, time.time()))
            sub_features = sub_feature_dict.setdefault(i[3], get_subreddit_features(i[3]))
            possible_input = np.concatenate((eval(i[0]), eval(i[1]), eval(i[2]), sub_features))
            possible_output = self.create_output_features(i)
            if self.validate(possible_input, possible_output):
                feature_list.append([possible_input, possible_output])
            else:
                invalid_inputs += 1
        print('invalid inputs:', invalid_inputs)
        testing_size = int(test_size*len(inputs))
        train_x = [i[0] for i in feature_list[testing_size:]]
        train_y = [i[1] for i in feature_list[testing_size:]]
        test_x = [i[0] for i in feature_list[:testing_size]]
        test_y = [i[1] for i in feature_list[:testing_size]]
        return train_x, train_y, test_x, test_y

    def create_output_features(self, output):
        output_array = np.zeros(num_of_score_buckets) #[0 for i in range(num_of_score_buckets)]
        for i in range(len(self.border_values_for_upvotes)):
            if output[4] < self.border_values_for_upvotes[i]:
                output_array[i] = 1
                return output_array
        output_array[len(self.border_values_for_upvotes)] = 1
        return output_array

    #build ngrams and rank comments
    #for each comment, remove stopwords then place into n-grams
    #sort the ngrams by how common they are, store the most common ones
    def read_metadata(self, get_preprocessed = True):
        inputs = get_db_input()
        score_list = []
        for i in inputs:
            score_list.append(i[4])
        self.border_values_for_upvotes = get_border_values(num_of_score_buckets, score_list)#gets the border of the buckets for the output features
        self.bucket_avg = create_average_of_buckets(self.border_values_for_upvotes, score_list)
        self.save_metadata()

    def save_model(self):
        self.saver.save(self.sess, model_save_location)
        logger.info('model saved')

    def load_model(self, saver, sess):
        saver.restore(sess, model_save_location)
        self.load_metadata()

    def save_metadata(self):
        with open('models/comment_classifier_upvote_border_values.pickle', 'wb') as f2:
            pickle.dump(self.border_values_for_upvotes, f2, protocol=pickle.HIGHEST_PROTOCOL)
        with open('models/comment_classifier_upvote_bucket.pickle', 'wb') as f2:
            pickle.dump(self.bucket_avg, f2, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info('metadata saved')

    def load_metadata(self):
        with open('models/comment_classifier_upvote_border_values.pickle', 'rb') as f2:
            self.border_values_for_upvotes = pickle.load(f2)
        with open('models/comment_classifier_upvote_bucket.pickle', 'rb') as f2:
            self.bucket_avg = pickle.load(f2)

#Feature creation methods:
#features of hour will be most useful followed by weekday and month, week of month is there for now
def create_input_feature_from_text(title, parent_text, child_text, title_time_stamp, parent_time_stamp, child_time_stamp, subreddit, subreddit_list, topic_models):
    child_features = preprocess_item(child_text, child_time_stamp, topic_models)
    parent_item = preprocess_item(parent_text, parent_time_stamp, topic_models)
    title_item = preprocess_item(title, title_time_stamp, topic_models)
    subreddit_features = get_subreddit_features(subreddit)
    return child_features + parent_item + title_item + subreddit_features

def preprocess_item(text, timestamp, topic_models):
    tokenized_comment = clean_and_tokenize(text)
    topic_features = []
    for i in topic_models:
        topic_features += get_topic_classification(tokenized_comment, i, tokenized=True)
    word_count_feature = get_word_count_feature(tokenized_comment)
    word_length_feature = get_word_length_feature(tokenized_comment)
    timestamp_features = create_timestamp_features(timestamp)
    pos_features = get_pos_tag_features(text)
    #print(len(topic_features), len(word_count_feature), len(word_length_feature), len(timestamp_features), len(pos_features))
    #print('here')
    return topic_features + timestamp_features + word_count_feature + word_length_feature + pos_features

def validate(input_len, input_features, out_put_features):
    if len(input_features) == input_len:
        return True
    else:
        return False

def get_topic_classification(text, topic_model, tokenized = False):
    #results = topic_memoization.setdefault((tuple(text), tokenized), topic_model.get_topic(tuple(text), tokenized=tokenized))
    results =  topic_model.get_topic(tuple(text), tokenized=tokenized)
    result_vector = [0 for i in range(topic_model.num_of_topics + 1)]
    results_sorted = sorted(results, key=lambda x: x[1], reverse = True)
    if len(results) == 0:
        result_vector[-1] = 1
    else:
        result_vector[results_sorted[0][0]] = 1
    return result_vector

def create_input_features(i, topic_models):
    return create_input_feature_from_text( i[21], i[14], i[5], i[24], i[7], i[16], i[2], topic_models)

def store_preprocessed_comment(comment_id, comment_features):
    pass

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
    return hour_feature + week_day_feature + week_month_feature + month_feature

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

def get_subreddit_features(s_id):
    subreddit_list = get_subreddit_list()
    subreddit_features =  [0 for i in range(len(subreddit_list))]
    subreddit_features[subreddit_list.index(s_id)] = 1
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

def get_pos_tag_features(input_text):
    tokens = nltk.tokenize.word_tokenize(input_text)
    pos_list = nltk.pos_tag(tokens)
    pos_features = [0 for i in range(len(pos_tag_map.keys())+ 1)]
    for i in pos_list:
        pos_features[pos_tag_map.get(i[1], 0)] = 1
    return pos_features

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

def get_word_length_feature(word_tokens):
    if len(word_tokens) == 0:
        return [0 for _ in range(max_word_length_for_features+1)]

    average_length = int(sum([len(i) for i in word_tokens])/len(word_tokens))
    features = [0 for _ in range(max_word_length_for_features+1)]
    if average_length < max_word_length_for_features:
        features[average_length] = 1
    else:
        features[-1] = 1
    return features

def get_word_count_feature(word_tokens):
    features = [0 for _ in range(max_word_count_for_features+1)]

    if len(word_tokens)//word_count_bucket_size < max_word_count_for_features:
        features[len(word_tokens)//word_count_bucket_size] = 1
    else:
        features[-1] = 1
    return features

def softmax(x):
    adj_x = []
    for i in x:
        if i >= 0:
            adj_x.append(i)
        else:
            adj_x.append(0)
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex


#get parent, child, post data from db
#allows user to only allow data from certai subreddits by passing list of elegible subreddit ids into it
def get_db_input():
    with sqlite3.connect('reddit.db') as conn:
        res = conn.execute('''select b.input, d.input, f.input, a.s_id, a.score
    from comments a join preprocessed_comments b on a.c_id = b.c_id
    join comments c on a.parent_id = c.c_id
    join preprocessed_comments d on c.c_id = d.c_id
    join posts e on a.p_id = e.p_id
    join preprocessed_posts f on e.p_id = f.p_id
    order by a.submitted_timestamp desc''').fetchall()
    logger.info('len of input = {0}'.format(len(res)))
    random.shuffle(res)
    output =  res[:max_results_to_analyze]
    return output

def get_db_input_to_pd():
    with sqlite3.connect('reddit.db') as conn:
        df = pd.read_sql_query('''select b.input, d.input, f.input, a.s_id, a.score
    from comments a join preprocessed_comments b on a.c_id = b.c_id
    join comments c on a.parent_id = c.c_id
    join preprocessed_comments d on c.c_id = d.c_id
    join posts e on a.p_id = e.p_id
    join preprocessed_posts f on e.p_id = f.p_id
    order by a.submitted_timestamp desc''', conn)
    logger.info('len of input = {0}'.format(df.size))

    if df.size < max_results_to_analyze:
        return df.sample(frac = 1)
    else:
        return df.sample(max_results_to_analyze)

def preprocess_all_comments(topic_models):
    with sqlite3.connect('reddit.db') as conn:
        res = conn.execute('''select c_id, body, submitted_timestamp
        from comments ''').fetchall()
        for count, i in enumerate(res):
            if count %10000 == 0:
                conn.commit()
                logger.info('{0} comments preprocessed'.format(count))
                print(i)
            try:
                conn.execute('insert into preprocessed_comments values (?, ?)', (i[0], repr(preprocess_item(i[1], i[2], topic_models))))
            except sqlite3.IntegrityError:
                conn.execute('update preprocessed_comments set input = ? where c_id = ?', (repr(preprocess_item(i[1], i[2], topic_models)), i[0]))
        conn.commit()

def preprocess_unprocessed_comments(topic_models):
    with sqlite3.connect('reddit.db') as conn:
        res = conn.execute('''select c_id, body, submitted_timestamp
        from comments where c_id not in (select c_id from preprocessed_comments)''').fetchall()
        for count, i in enumerate(res):
            if count %10000 == 0:
                conn.commit()
                logger.info('{0} comments preprocessed'.format(count))
            try:
                conn.execute('insert into preprocessed_comments values (?, ?)', (i[0], repr(preprocess_item(i[1], i[2], topic_models))))
            except sqlite3.IntegrityError:
                conn.execute('update preprocessed_comments set input = ? where c_id = ?', (repr(preprocess_item(i[1], i[2], topic_models)), i[0]))
        conn.commit()

def preprocess_all_posts(topic_models):
    with sqlite3.connect('reddit.db') as conn:
        res = conn.execute('''select p_id, title, timestamp
        from posts ''').fetchall()
        for count, i in enumerate(res):
            if count %10000== 0:
                conn.commit()
                logger.info('{0} posts preprocessed'.format(count))
            try:
                conn.execute('insert into preprocessed_posts values (?, ?)', (i[0], repr(preprocess_item(i[1], i[2], topic_models))))
            except sqlite3.IntegrityError:
                conn.execute('update preprocessed_posts set input = ? where p_id = ?', (repr(preprocess_item(i[1], i[2], topic_models)), i[0]))
        conn.commit()

def preprocess_unprocessed_posts(topic_models):
    with sqlite3.connect('reddit.db') as conn:
        res = conn.execute('''select p_id, title, timestamp
        from posts where p_id not in (select p_id from preprocessed_posts)''').fetchall()
        for count, i in enumerate(res):
            if count %10000== 0:
                conn.commit()
                logger.info('{0} posts preprocessed'.format(count))
            try:
                conn.execute('insert into preprocessed_posts values (?, ?)', (i[0], repr(preprocess_item(i[1], i[2], topic_models))))
            except sqlite3.IntegrityError:
                conn.execute('update preprocessed_posts set input = ? where c_id = ?', (repr(preprocess_item(i[1], i[2], topic_models)), i[0]))
        conn.commit()

def get_subreddit_list():
    global subreddit_list
    if len(subreddit_list) == 0:
        with sqlite3.connect('reddit.db') as conn:
            res = conn.execute('select distinct s_id from subreddits order by s_id').fetchall()
            subreddit_list = [i[0] for i in res]
    return subreddit_list

#testing
if __name__ == '__main__':
    topics = []
    for i in [10, 50, 100, 200]:
        topics.append(Reddit_LDA_Model(i))

    preprocess_unprocessed_comments(topics)
    preprocess_unprocessed_posts(topics)
    dnn = DNN_comment_classifier(topics, retrain=True)
    dnn.train_nn(100)
    print('here')



