import bot
from sentiment_classifier import DNN_sentiment_classifier
from comment_success_classifier import DNN_comment_classifier

if __name__ == '__main__':
    sentiment_model = DNN_sentiment_classifier()
    comment_classifier = DNN_comment_classifier()
    comment_classifier.train_nn(10, sentiment_model)