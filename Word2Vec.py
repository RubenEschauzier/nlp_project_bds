import json
import logging, os
import pickle
import string
import tempfile

from gensim.models import Word2Vec
from nltk import tokenize
import gensim
import gensim.downloader as api
import numpy as np

# https://datascience.stackexchange.com/questions/42157/updating-google-news-word2vec-word-embedding

class MyCorpus(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for x, line in enumerate(open(self.filename, errors='ignore')):
            text = tokenize.sent_tokenize(json.loads(line)['input_text'])
            for sentence in text:
                yield sentence.translate(str.maketrans('', '', string.punctuation)).lower().split()


def save_model(model):
    with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
        temporary_filepath = 'Word2VecModel'
        model.save(temporary_filepath)


def load_model(filelocation):
    model = gensim.models.Word2Vec.load(filelocation)
    print(model.wv.most_similar(positive='food'))
    return model


def transfer_learning_model():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    pre_trained_model = api.load('word2vec-google-news-300')
    sentences = [['Here', 'is', 'another', 'sentence']]
    model = Word2Vec(size = 300, window = 5, min_count = 10, workers = 2)
    model.intersect_word2vec_format(pre_trained_model, lockf=1.0, binary=True)
    model.train(sentences, total_examples=model.corpus_count, epochs = 5)


if __name__ == '__main__':
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # sentences = MyCorpus('review-003.json')
    # model = gensim.models.Word2Vec(sentences=sentences, size=300, min_count=10)
    # save_model(model)
    #load_model('Word2VecModel')
    transfer_learning_model()