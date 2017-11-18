import bot
from sentiment_classifier_naive_bayes import NBsentimentClassifier
from topic_model import Reddit_LDA_Model
from comment_success_classifier import DNN_comment_classifier
import logging
import sqlite3
import random
import datetime
import numpy as np
from data_manager import Reddit_Memoization

data_memoization = Reddit_Memoization()

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#not all comments have a hgih value responce, checks a sample fo comments per post and then moves on if it does nto see an ideal post
top_values_returned = []
min_comment_success_threashold = .95
max_comments_to_sample_from = 250000
comments_to_consider_per_post = 10000
comment_pool = []

def get_comment_results(comment_classifier, parent_text, parent_time_stamp, title, title_time_stamp, subreddit, child_text, child_time_stamp, sentiment_classifier, topic_model):
    return comment_classifier.create_input_feature_from_text(title, parent_text, child_text, title_time_stamp, parent_time_stamp, child_time_stamp, subreddit, sentiment_classifier, topic_model)

def load_comment_pool():
    global comment_pool
    with sqlite3.connect('reddit.db') as conn:
        res = conn.execute('select body from comments').fetchall()
        for r in res[0:max_comments_to_sample_from]:
            comment_pool.append(r[0])

def get_best_comment(post_info, comment_classifier, nb_sentiment_model, lda_topic_model):
    sorting_struct = []
    small_comment_pool = random.sample(comment_pool, comments_to_consider_per_post)
    for i in small_comment_pool:
        sorting_struct.append((i, get_comment_results(comment_classifier, i['parent_body'],
                                                      i['parent_timestamp'], i['post_title'],
                                                      i['post_timestamp'], i['s_id'],
                                                      i, datetime.datetime.now().timestamp(),
                                                      nb_sentiment_model, lda_topic_model)))
    sorting_struct.sort(key=lambda i: i[1][-1])

def evaluate_comments(comment_classifier, nb_sentiment_model, lda_topic_model):
    global top_values_returned
    time_of_last_comment_posted = datetime.datetime.now().timestamp()

    for i in bot.get_comments_for_most_recent_posts():
        best_comment_for_reply = get_best_comment(i, comment_classifier, nb_sentiment_model, lda_topic_model)
        top_values_returned.append(best_comment_for_reply[1][-1])
        if datetime.datetime.now().timestamp():
            pass

def main(train_comment_classifier = False):
    nb_sentiment_model = NBsentimentClassifier(data_memoization)
    lda_topic_model = Reddit_LDA_Model(data_memoization)

    if train_comment_classifier:
        comment_classifier = DNN_comment_classifier(data_memoization, retrain = True)
        comment_classifier.train_nn(10, nb_sentiment_model, lda_topic_model)
    bot.main()


if __name__ == '__main__':
    main(train_comment_classifier=True)
