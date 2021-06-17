import itertools
from tqdm import tqdm
import string
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import nltk
import re
import gensim.corpora as corpora
import gensim
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from wordcloud import WordCloud
import gensim.downloader as api
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def load_data(file_loc, type_data='question') -> np.array:
    """
    :param file_loc: Location of raw pickled question_text file
    :return: text_array: numpy array of raw questions
    """
    with open(file_loc, 'rb') as f:
        data = pickle.load(f)
    if type_data == 'question':
        text_array = np.array(list(data.keys()))
        for i in range(text_array.size):
            text_array[i] = text_array[i].replace('-', " ").lower()
        return pd.DataFrame(text_array, columns=['Question'])
    else:
        return data


def annotate_data(answer_data, num_to_annotate):
    annotated_answers = []
    annotations = []
    to_annotate = answer_data.sample(n=num_to_annotate)
    for i, answer in enumerate(to_annotate.iterrows()):
        print('Answer {}/{}'.format(i, num_to_annotate))
        ask_input = True
        while ask_input:
            label = int(input("Please annotate: {}".format('\n'.join(sent_tokenize(answer[1]['Answer'])))))
            if label == 1 or label == 0 or label == -1:
                annotated_answers.append(answer[0])
                annotations.append(label)
                ask_input = False
    to_annotate['Ground_Truth'] = annotations
    print(to_annotate)
    to_annotate.to_csv('data/annotated_dataframe')
    np.save('data/annotated_answers', annotated_answers)
    np.save('data/annotations', annotations)


def clean_answers(answer_data):
    answer_data['Answer'] = answer_data['Answer'].apply(
        lambda x: x.split("Continue Reading")[len(x.split("Continue Reading")) - 1])
    return answer_data


def get_wordnet_pos(treebank_tag):
    """
    return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # As default pos in lemmatization is Noun
        return wordnet.NOUN


def remove_stopwords(input_text) -> list:
    """
    :param input_text: input list of tokenized questions
    :return: question_text: output list of tokenized questions with stopwords removed
    """
    stop_words = stopwords.words('english')
    for i in range(len(input_text)):
        input_text[i] = ([word for word in input_text[i] if word.lower() not in stop_words])

    return input_text


def tokenize_text(input_text) -> list:
    """
    :param input_text: numpy array of raw question question_text to be converted into tokens
    :return: output_text: list of tokenized questions
    """
    output_text = []

    for i in range(input_text.size):
        output_text.append(word_tokenize(re.sub('[' + string.punctuation + ']', '', input_text[i])))
    return output_text


def get_cosine_similarity(x, y):
    return ((np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))) + 1) / 2


def create_wordcloud(data):
    long_string = ','.join(list(data))
    wordcloud = WordCloud(background_color="black", max_words=5000, contour_width=10, contour_color='steelblue')
    wordcloud.generate(long_string)
    plt.figure()
    # plot words
    plt.imshow(wordcloud, interpolation="bilinear")
    # remove axes
    plt.axis("off")
    # show the result
    plt.show()
    plt.show()

    word_freq = {}
    word_list = long_string.split(' ')
    stop_words = stopwords.words('english')
    word_list = ([word for word in word_list if word not in stop_words])

    for w in word_list:
        if w not in word_freq.keys():
            word_freq[w] = word_list.count(w)
    print('Total amount of unique words in the corpus: {}'.format(len(word_freq)))
    sorted_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
    sorted_freq = dict(itertools.islice(sorted_freq.items(), 50))
    plt.bar(list(sorted_freq.keys()), sorted_freq.values(), color='g')
    plt.xticks(rotation='vertical')
    plt.margins(0.01)
    plt.subplots_adjust(bottom=0.2)
    plt.show()


def descriptive_statistics(data):
    total_words = 0
    total_question = 0
    for question in data:
        total_words += len(question)
        total_question += 1
    print('The total amount of words in the corpus: {}'.format(total_words))
    print('The total amount of questions in the corpus: {}'.format(total_question))


def get_time():
    return datetime.now().strftime("%H:%M:%S")


def lemmatize_text(input_text):
    lemmatizer = WordNetLemmatizer()
    apply_lemmatization = lambda x: [lemmatizer.lemmatize(word[0], get_wordnet_pos(word[1])) for word in x]
    tagged_text = [nltk.pos_tag(text) for text in input_text]
    lemmatized_text = [apply_lemmatization(text) for text in tagged_text]
    return lemmatized_text


if __name__ == '__main__':
    # https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
    # For LDA https://stats.stackexchange.com/questions/349761/reasonable-hyperparameter-range-for-latent-dirichlet-allocation
    question_text = load_data('data_prelim/questions.p')
    answers = load_data('data_prelim/answers.p', type_data='answer')
    answers = clean_answers(answers)
    answer_text = answers['Answer'].values

    tokenized_answers = tokenize_text(answer_text)
    processed_answers = remove_stopwords(tokenized_answers)
    lemmatized_answers = lemmatize_text(processed_answers)

    main_categorise(lemmatized_answers, answers, lemmatized_answers, answer_text)

    # tokenized_text = tokenize_text(question_text)
    # processed_text = remove_stopwords(tokenized_text)
    # lemmatized_text = lemmatize_text(processed_text)
    # LDA_scikit(lemmatized_text)

    # Uncomment this to get descriptive statistics
    # descriptive_statistics(processed_text)
    # create_wordcloud(question_text)
