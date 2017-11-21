import bot
from sentiment_classifier_naive_bayes import NBsentimentClassifier
from topic_model import Reddit_LDA_Model
from comment_success_classifier import DNN_comment_classifier
import logging
import sqlite3
import random
import datetime
import time

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#not all comments have a hgih value responce, checks a sample fo comments per post and then moves on if it does nto see an ideal post
top_values_returned = []
comments_to_consider_per_iteration = 10000
max_comments_to_sample_from = 250000
comment_pool = []
num_of_topics = 100
num_of_topics_recorded = 5 #records up to top 5 probable topics
banned_words = [' bot', 'https']


#faster than dealing with db io
def load_comment_pool():
    global comment_pool
    with sqlite3.connect('reddit.db') as conn:
        res = conn.execute('select body from comments').fetchall()
        for r in res[0:max_comments_to_sample_from]:
            eligible = True
            for b in banned_words:
                if b in r[0].lower():
                    eligible = False
            if eligible:
                comment_pool.append(r[0])

#just remapping for simplicity
def get_comment_results(comment_classifier, sentiment_classifier, topic_model, parent_text, parent_time_stamp, title, title_time_stamp, subreddit, child_text, child_time_stamp):
    input_features = comment_classifier.create_input_feature_from_text(title, parent_text, child_text, title_time_stamp, parent_time_stamp, child_time_stamp, subreddit, sentiment_classifier, topic_model)
    return comment_classifier.run_input(input_features)

#tries comments_to_consider_per_iteration number of possible comment parent combinations, returns most likely to get top bucket comment details
def evaluate_comments(comment_classifier, nb_sentiment_model, lda_topic_model):
    possible_comments_to_reply_to = bot.get_comments_for_most_recent_posts()
    sorting_struct = []
    start_time = time.time()

    for count, i in enumerate(range(comments_to_consider_per_iteration)):
        if count%1000 == 0:
            logger.info('considered {0} comments, time: {1}'.format(count , time.time()-start_time))
        possible_comment = random.choice(comment_pool)
        possible_post = random.choice(possible_comments_to_reply_to)
        result = get_comment_results(comment_classifier, nb_sentiment_model, lda_topic_model, possible_post['parent_body'],
                            possible_post['parent_timestamp'], possible_post['post_title'], possible_post['post_timestamp'],
                            possible_post['s_id'], possible_comment, str(datetime.datetime.now().timestamp()))
        sorting_struct.append({'parent_id':possible_post['parent_id'], 'post_details':possible_post, 'possible_comment_text':possible_comment, 'classifier_results':result})

    sorting_struct.sort(key=lambda i: i['classifier_results'], reverse = True)
    logger.info('top picks: ')
    for i in sorting_struct[0:5]:
        print(i)
    return sorting_struct[0]

def main(train_comment_classifier = False):
    nb_sentiment_model = NBsentimentClassifier()
    lda_topic_model = Reddit_LDA_Model(num_of_topics)
    load_comment_pool()

    if train_comment_classifier:
        comment_classifier = DNN_comment_classifier(num_of_topics, num_of_topics_recorded)
        #comment_classifier = DNN_comment_classifier(num_of_topics, num_of_topics_recorded, retrain = True)
        #comment_classifier.train_nn(8, nb_sentiment_model, lda_topic_model)
    while True:
        #bot.scrape_one_iteration()
        #comment_to_post = evaluate_comments(comment_classifier, nb_sentiment_model, lda_topic_model)
        #print(comment_to_post)
        #bot.post_reply(comment_to_post['parent_id'], comment_to_post['possible_comment_text'])
        break
    input_1= comment_classifier.create_input_feature_from_text("Donald Trump is shutting down his charitable foundation",
                                                               "his money laundering company is being shut down in a panic you say? they're burning everything you say?",
                                                               "Businesses are required to retain their tax documents by law for 6 years, even if they fold.",
                                                               1511225948, 1511225948, 1511225948, 't5_2cneq', nb_sentiment_model, lda_topic_model)
    print(input_1)
    print(comment_classifier.run_input(input_1))


if __name__ == '__main__':
    main(train_comment_classifier=True)
