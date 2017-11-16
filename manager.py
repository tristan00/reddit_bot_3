import bot
from sentiment_classifier_naive_bayes import NBsentimentClassifier
from comment_success_classifier import DNN_comment_classifier

if __name__ == '__main__':
    sentiment_model = NBsentimentClassifier()
    comment_classifier = DNN_comment_classifier()
    comment_classifier.train_nn(10, sentiment_model)
    bot.main()