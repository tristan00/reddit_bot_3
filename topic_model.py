import gensim
import nltk
import string
import sqlite3
import traceback
import logging

stop_word_set = set(nltk.corpus.stopwords.words('english'))
lda_model_location = '/models/lda_model.txt'
corpus_location = '/models/serialized_corpus.mm'
dictionary_location = "models/tag_dictionary_lda.pkl"

class Reddit_LDA_Model():
    def __init__(self, num_of_topics):
        self.build_lda(num_of_topics)
        result = self.get_topic("Trump and his allies may have colluded with Russia for different reasons, nevertheless a pattern seems to be emerging in which a conspiracy to commit offense or to defraud the United States becomes more likely on every passing day as more information comes to light.")
        print(result)


    #TODO: make generator out of underlying db input function
    def get_texts(self):
        for input in get_db_input():
            output = clean_and_tokenize(input)
            yield output

    def get_corpus(self):
        self.dictionary = gensim.corpora.Dictionary(self.get_texts())
        self.dictionary.save(dictionary_location)
        corpus = [self.dictionary.doc2bow(text) for text in self.get_texts()]
        gensim.corpora.MmCorpus.serialize(corpus_location, corpus)
        return corpus

    #try and load, if not it builds
    def build_lda(self, num_of_topics):
        try:
            self.corpus = gensim.corpora.MmCorpus(corpus_location)
            self.dictionary = gensim.corpora.Dictionary.load(dictionary_location)
            self.lda = gensim.models.ldamodel.LdaModel.load(lda_model_location)
        except:
            traceback.print_exc()
            logging.info('training topic model:')
            self.corpus = self.get_corpus()
            self.lda = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=num_of_topics)
            self.lda.save(lda_model_location)

    def get_topic(self, text, minimum_probability = .1, tokenized=False):
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
        return [i[0] for i in res]

if __name__ == '__main__':
    corpus_class = Reddit_LDA_Model()


