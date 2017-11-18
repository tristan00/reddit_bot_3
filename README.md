# reddit_bot_3, #2 had repo issues, #1 sucked

Organization:
bot.py: praw based bot that scrapes reddit and will have the ability to post

manager.py: runs models and directs bot to do actions.

comment_success_classifier: dnn to classify comments by success. It uses ngrams, timestamps, subreddit and sentiment of the post title,
the parent comment and the child/considered comment. DNNS are not ideal for generating text but with a large db of comments most
comments that would be ideal to say are already somewhere in the db. Long term this should be switched out for a proper generative model.

sentiment_classifier.py: basic sentiment classifier

bot_detection_classifier.py: Adverserial network trained to detect my bot. Manager will consider this input when picking comment.

topic_classifier.py: clustering, lsa, lda or other to classify content by topic, will be used as inputs to other models

TODO:

Build gensim topic model

Build more text analysis tools for use in models
-text complexity

build manager bot handling lgic

build generators for dictionary use, figure out how to avoid deadlocks (secondary dicts?)

figure out hw to remove dropouts for testing

build bot_detection_classifier

replace success classifier with generative model