import gensim
import nltk
import string
import sqlite3
import traceback
import logging
import random

stop_word_set = set(nltk.corpus.stopwords.words('english'))
lda_model_location = 'models/lda_model_{0}_topics.txt'
corpus_location = 'models/serialized_corpus_{0}.mm'
dictionary_location = "models/tag_dictionary_lda_{0}.pkl"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Reddit_LDA_Model():
    def __init__(self, num_of_topics):
        self.num_of_topics = num_of_topics
        self.build_lda()
        result = self.get_topic("Trump and his allies may have colluded with Russia for different reasons, nevertheless a pattern seems to be emerging in which a conspiracy to commit offense or to defraud the United States becomes more likely on every passing day as more information comes to light.")
        print(result)

    #TODO: make generator out of underlying db input function
    def get_texts(self):
        for input in get_db_input():
            output = clean_and_tokenize(input)
            yield output

    def get_corpus(self):
        self.dictionary = gensim.corpora.Dictionary(self.get_texts())
        self.dictionary.save(dictionary_location.format(self.num_of_topics))
        corpus = [self.dictionary.doc2bow(text) for text in self.get_texts()]
        gensim.corpora.MmCorpus.serialize(corpus_location.format(self.num_of_topics), corpus)
        return corpus

    #try and load, if not it builds
    def build_lda(self):
        try:
            self.corpus = gensim.corpora.MmCorpus(corpus_location.format(self.num_of_topics))
            self.dictionary = gensim.corpora.Dictionary.load(dictionary_location.format(self.num_of_topics))
            self.lda = gensim.models.ldamodel.LdaModel.load(lda_model_location.format(self.num_of_topics))
        except:
            traceback.print_exc()
            logging.info('training topic model:')
            self.corpus = self.get_corpus()
            self.lda = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.num_of_topics)
            self.lda.save(lda_model_location.format(self.num_of_topics))

    def get_topic(self, text, minimum_probability = .01, tokenized=False):
        if tokenized:
            text_bow = self.dictionary.doc2bow(text)
            return self.lda.get_document_topics(text_bow)
        else:
            text_bow = self.dictionary.doc2bow(clean_and_tokenize(text))
            return self.lda.get_document_topics(text_bow)

def clean_and_tokenize(input_text):
    clean_text = remove_punctuation_from_text(input_text.lower())
    return remove_stopwords_from_list(nltk.tokenize.word_tokenize(clean_text))

def remove_stopwords_from_list(input_list):
    results = []
    for i in input_list:
        if i not in stop_word_set:
            results.append(i)
    return results

def remove_punctuation_from_text(input_text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in input_text if ch not in exclude)

def get_db_input():
    with sqlite3.connect('reddit.db') as conn:
        res = conn.execute('''select body
    from comments''').fetchall()
        output = [i[0] for i in res]
        random.shuffle(output)
        return output

if __name__ == '__main__':
    print('n:', 10)
    corpus_class = Reddit_LDA_Model(10)
    print(corpus_class.lda.print_topics())

    print('n:', 50)
    corpus_class = Reddit_LDA_Model(50)
    corpus_class.lda.show_topics()

    print('n:', 100)
    corpus_class = Reddit_LDA_Model(100)
    corpus_class.lda.show_topics()

    print('n:', 200)
    corpus_class = Reddit_LDA_Model(200)
    corpus_class.lda.show_topics()



