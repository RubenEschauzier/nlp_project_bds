import operator
import numpy as np
import pandas as pd
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from datetime import datetime

from answer_filter import get_cosine_similarity, get_time


def create_document_vec(input_text):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(input_text)]
    model = Doc2Vec(documents, vector_size=300, window=8, min_count=2)
    return model


def create_topic_vectors():
    # These words were (mostly) compiled by Dr. Meike Morren.

    diet_words = ['diet', 'healthy', 'nutrition', 'vitamins', 'dietary', 'guidelines', 'adoption', 'live',
                  'source',
                  'longevity', 'lifestyle', 'artificial', 'lab', 'cultured', 'cellular', 'protein']
    # removed words vegan, vegetarian due to common occurence
    vega_words = ['pescatarian', 'milk', 'cheese', 'welfare', 'care',
                  'slaughter', 'cruelty', 'free', 'suffering', 'preservation', 'dairy']
    meat_words = ['meat', 'carnivores', 'omnivores', 'pork', 'veal', 'lamb', 'beef', 'lard', 'bacon', 'steak', 'ham',
                  'chop', 'sausage', 'sirloin', 'wings', 'ribeye', 'ribs', 'riblets']
    # Added recipe
    cook_words = ['baking', 'cooking', 'preparing', 'braising', 'restaurant', 'gourmet', 'supermarket', 'dining',
                  'roasting', 'recipe']
    animal_words = ['animals', 'flesh', 'fish', 'pig', 'chicken', 'lamb', 'sheep', 'cow', 'calf', 'goat', 'piglet',
                    'land', 'mammals', 'reptiles', 'birds', 'amphibians']

    word_vectors = gensim.downloader.load('word2vec-google-news-300')

    diet_words_vectors = [word_vectors[word] for word in diet_words]
    vega_words_vectors = [word_vectors[word] for word in vega_words]
    meat_words_vectors = [word_vectors[word] for word in meat_words]
    cook_words_vectors = [word_vectors[word] for word in cook_words]
    animal_words_vectors = [word_vectors[word] for word in animal_words]

    return diet_words_vectors, vega_words_vectors, meat_words_vectors, cook_words_vectors, animal_words_vectors


def categorise_answers(document_vectors, documents, n_answers, diet_v, vega_v, meat_v, cook_v, animal_v, answer_df):
    diet_docs = []
    vega_docs = []
    meat_docs = []
    cook_docs = []
    animal_docs = []
    answer_df['category_w2v'] = np.nan
    categories = []
    for i in range(n_answers):
        scores = {'diet_score': 0, 'vega_score': 0, 'meat_score': 0, 'cook_score': 0, 'animal_score': 0}
        for diet_vector in diet_v:
            scores['diet_score'] += get_cosine_similarity(diet_vector, document_vectors[i]) / len(diet_v)
        for vega_vector in vega_v:
            scores['vega_score'] += get_cosine_similarity(vega_vector, document_vectors[i]) / len(vega_v)
        for meat_vector in meat_v:
            scores['meat_score'] += get_cosine_similarity(meat_vector, document_vectors[i]) / len(meat_v)
        for cook_vector in cook_v:
            scores['cook_score'] += get_cosine_similarity(cook_vector, document_vectors[i]) / len(cook_v)
        for animal_vector in animal_v:
            scores['animal_score'] += get_cosine_similarity(animal_vector, document_vectors[i]) / len(animal_v)
        max_score = max(scores.items(), key=operator.itemgetter(1))[0]
        if max_score == 'diet_score':
            diet_docs.append(documents[i])
            categories.append(1)
        if max_score == 'vega_score':
            vega_docs.append(documents[i])
            categories.append(2)
        if max_score == 'meat_score':
            meat_docs.append(documents[i])
            categories.append(3)
        if max_score == 'cook_score':
            cook_docs.append(documents[i])
            categories.append(4)
        if max_score == 'animal_score':
            animal_docs.append(documents[i])
            categories.append(5)
    answer_df['category_w2v'] = categories
    print(
        '{}: Number in diet (manual): {}, number in vega: {}, number in meat: {}, number in cook: {}, number in animal: {}'.
            format(get_time(), len(diet_docs), len(vega_docs), len(meat_docs), len(cook_docs), len(animal_docs)))

    return diet_docs, vega_docs, meat_docs, cook_docs, animal_docs, answer_df


def categorise_questions(vega_v, meat_v, document_vectors, question_df, n_answers):
    question_df['category_w2v'] = np.nan
    categories = []
    num_vega = 0
    num_meat = 0
    for i in range(n_answers):
        scores = {'vega_score': 0, 'meat_score': 0, }
        for vega_vector in vega_v:
            scores['vega_score'] += get_cosine_similarity(vega_vector, document_vectors[i]) / len(vega_v)
        for meat_vector in meat_v:
            scores['meat_score'] += get_cosine_similarity(meat_vector, document_vectors[i]) / len(meat_v)

        max_score = max(scores.items(), key=operator.itemgetter(1))[0]

        if max_score == 'vega_score':
            categories.append(0)
            num_vega += 1
        if max_score == 'meat_score':
            categories.append(1)
            num_meat += 1
    question_df['category_w2v'] = categories
    print('{}: Number in vega: {}, number in meat: {}'.format(get_time(), num_vega, num_meat))

    return question_df

# def statistics_categories_manual(answer_dfd_doc, v_doc, m_doc, c_doc, a_doc, question_df, raw_answers):
#     # Most inefficient code in the world unfortunately, however it works so..
#     stop_words = stopwords.words('english')
#     categories = [d_doc, v_doc, m_doc, c_doc, a_doc]
#
#     for i, category in enumerate(categories):
#         sentence_counts = []
#         word_counts = []
#         word_freq = {}
#         print('{}: Creating List of words'.format(get_time()))
#         word_list = [word for document in category for word in document]
#         print('{}: Counting words'.format(get_time()))
#
#         for w in word_list:
#             if w not in word_freq.keys():
#                 word_freq[w] = word_list.count(w)
#
#         print('{}: Getting sentence lengths'.format(get_time()))
#         # Change to take raw text
#         print(raw_answers.shape)
#         raw_answers2 = question_df['Answer'][question_df['category_w2v'] == i + 1]
#         print(raw_answers2.shape)
#         for document in raw_answers2:
#             document.translate(dict.fromkeys(string.punctuation))
#             words = word_tokenize(document)
#             word_tokenized = ([word for word in words if word not in stop_words])
#             sentences = sent_tokenize(document)
#             sentence_counts.append(len(sentences))
#             word_counts.append(len(word_tokenized))
#
#         print('Sorting lengths')
#         sorted_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
#         sorted_freq = dict(itertools.islice(sorted_freq.items(), 10))
#         sentence_counts = np.array(sentence_counts)
#         word_counts = np.array(word_counts)
#         print('For category: {} (manual), The top 10 most frequent words are: {}'.format(i, sorted_freq))
#         print('Total amount of unique words in the category manual: {}'.format(len(word_freq)))
#         print('Mean sentences per document: {}, std: {}, median: {}, min: {}, max: {}'.format(np.mean(sentence_counts),
#                                                                                               np.std(sentence_counts),
#                                                                                               np.median(
#                                                                                                   sentence_counts),
#                                                                                               np.min(sentence_counts),
#                                                                                               np.max(sentence_counts)))
#         print('Mean words per document: {}, std: {}, median: {}, min: {}, max: {}'.format(np.mean(word_counts),
#                                                                                           np.std(word_counts),
#                                                                                           np.median(word_counts),
#                                                                                           np.min(word_counts),
#                                                                                           np.max(word_counts)))
