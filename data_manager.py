import nltk

stop_word_list = list(nltk.corpus.stopwords.words('english'))
stop_word_set = set(stop_word_list)
import string

punc_trans = str.maketrans('', '', string.punctuation)

class Reddit_Memoization():
    def __init__(self):
        self.tokenization_transformation = {}

    def clean_and_tokenize(self, input_text):
        return self.tokenization_transformation.setdefault(input_text, clean_and_tokenize(input_text))

    def get_clean_text(self, input_text):
        return ' '.join(self.tokenization_transformation.setdefault(input_text, clean_and_tokenize(input_text)))

#sqlite generator
def sqlite_generator():
    pass

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
    return input_text.translate(punc_trans)

def remove_stopwords_from_list(input_list):
    results = []
    for i in input_list:
        if i not in stop_word_set:
            results.append(i)
    return results
