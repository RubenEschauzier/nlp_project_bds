import json
import re
import string

import numpy as np
import pandas as pd
from gensim import corpora
from nltk import word_tokenize, sent_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV

from answer_filter import remove_stopwords, tokenize_text, get_time


def create_input_lda(input_text):
    """
    Converts question_text into format needed to apply LDA
    :param input_text: Input tokenized sentences as a list of lists
    :return: word_dict: A dictionary mapping unique word_ids with words and
             corpus: The question_text converted into the number of occurences of word_ids.
    """
    word_dict = corpora.Dictionary(input_text)
    corpus = [word_dict.doc2bow(doc) for doc in input_text]
    return word_dict, corpus

def show_topics(vectorizer, lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


def LDA_scikit_gridsearch(lemma_text):
    input_count_vec = [' '.join(x) for x in lemma_text]
    vectorizer = CountVectorizer()
    model_input = vectorizer.fit_transform(input_count_vec)

    params = {'n_components': [6, 8, 10, 12, 14, 16]}
    params_quick = {'n_components': [6, 8]}
    model = LatentDirichletAllocation(random_state=2021)
    lda_search = GridSearchCV(model, param_grid=params, n_jobs=-1, refit=True)
    lda_search.fit(model_input)
    best_lda_model = lda_search.best_estimator_
    print("Best Model's Params: ", lda_search.best_params_)
    print("Best Log Likelihood Score: ", lda_search.best_score_)
    print("Model Perplexity: ", best_lda_model.perplexity(model_input))

    json.dump(lda_search.best_params_, open("lda_output/best_params_lda", 'w'))

    lda_output = best_lda_model.transform(model_input)
    dominant_topics = np.argmax(lda_output, axis=1)
    unique_elements, counts_elements = np.unique(dominant_topics, return_counts=True)
    print("Total number of documents in each topic: {}".format(counts_elements))

    topic_keywords = show_topics(vectorizer, lda_model=best_lda_model, n_words=15)

    # Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
    print(df_topic_keywords)
    df_topic_keywords.to_csv('lda_output/topic_keywords')

# Still implement automatic loading of best parameters
def divide_documents(lemma_text, n_topics, alpha, beta, answer_df, load_best_params = False):
    if load_best_params:
        best_params = json.load(open("lda_output/best_params_lda.json"))

    input_count_vec = [' '.join(x) for x in lemma_text]
    vectorizer = CountVectorizer()
    model_input = vectorizer.fit_transform(input_count_vec)

    lda_model = LatentDirichletAllocation(n_components=n_topics).fit(
        model_input)
    lda_output = lda_model.transform(model_input)
    dominant_topics = np.argmax(lda_output, axis=1)

    return dominant_topics


def statistics_categories_LDA(answer_df, category_column='LDA_topics', col_name = 'Answer', num_topics = 6):
    assert category_column in answer_df.columns.values

    for i in range(num_topics):
        sentence_counts = []
        word_counts = []

        print('{}: Selecting relevant text and tokenizing'.format(get_time()))
        text = answer_df[col_name][answer_df[category_column] == i].values
        if len(text) > 0:
            word_tokenized = tokenize_text(text)
            word_tokenized = remove_stopwords(word_tokenized)

            print('{}: Getting sentence lengths'.format(get_time()))
            for document in text:
                sentences = sent_tokenize(document)
                sentence_counts.append(len(sentences))

            print('{}: Counting words'.format(get_time()))
            for tokenized_document in word_tokenized:
                word_counts.append(len(tokenized_document))

            print('Mean sentences per document(LDA): {}, std: {}, median: {}, min: {}, max: {}'.format(
                np.mean(sentence_counts),
                np.std(sentence_counts),
                np.median(sentence_counts),
                np.min(sentence_counts),
                np.max(sentence_counts)))
            print('Mean words per document(LDA): {}, std: {}, median: {}, min: {}, max: {}'.format(np.mean(word_counts),
                                                                                                   np.std(word_counts),
                                                                                                   np.median(word_counts),
                                                                                                   np.min(word_counts),
                                                                                                   np.max(word_counts)))
