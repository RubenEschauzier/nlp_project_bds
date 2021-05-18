import numpy as np
import pickle
import nltk
import gensim.corpora as corpora
import gensim
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import gensim.downloader as api
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def load_data(file_loc, type='question') -> np.array:
    """
    :param file_loc: Location of raw pickled question_text file
    :return: text_array: numpy array of raw questions
    """
    with open(file_loc, 'rb') as f:
        data = pickle.load(f)
    if type == 'question':
        text_array = np.array(list(data.keys()))
        for i in range(text_array.size):
            text_array[i] = text_array[i].replace('-', " ").lower()
        return text_array
    else:
        return data


def annotate_data(answer_data, num_to_annotate):
    annotated_answers = []
    annotations = []
    to_annotate = answer_data.sample(n=num_to_annotate)
    for row in to_annotate:
        ask_input = True
        while ask_input:
            label = input("Please annotate: {}".format(row['Answer']))
            if label == 1 or label == 0 or label == -1:
                annotated_answers.append(row['Answer'])
                annotations.append(label)
                ask_input = False
    np.save('data/annotated_answers', annotated_answers)
    np.save('data/annotations', annotations)

def clean_answers(answer_data):
    answer_data['Answer'] = answer_data['Answer'].apply(lambda x: x.split("Continue Reading")[len(x.split("Continue Reading"))-1])
    return answer_data

def estimate_sentiment(text_data):
    total_scores = []
    binary_score = []
    for answer in text_data['Answer']:
        answer_score = 0
        sent_answer = nltk.sent_tokenize(answer)
        for sentence in sent_answer:
            analyser = SentimentIntensityAnalyzer()
            score = analyser.polarity_scores(sentence)
            answer_score += score['compound']
        total_scores.append(answer_score)
        if answer_score > 0.05:
            binary_score.append(1)
        else:
            binary_score.append(-1 if answer_score < 0.05 else 0)
    text_data['binary_score'] = binary_score
    text_data['score'] = total_scores
    print(text_data)
    return text_data


def remove_stopwords(input_text) -> list:
    """
    :param input_text: input list of tokenized questions
    :return: question_text: output list of tokenized questions with stopwords removed
    """
    stop_words = stopwords.words('english')
    for i in range(len(input_text)):
        input_text[i] = ([word for word in input_text[i] if word not in stop_words])

    return input_text


def tokenize_text(input_text) -> list:
    """
    :param input_text: numpy array of raw question question_text to be converted into tokens
    :return: output_text: list of tokenized questions
    """
    output_text = []
    for i in range(input_text.size):
        output_text.append(word_tokenize(input_text[i]))
    return output_text


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


def apply_lda(word_dict, corpus, num_topics):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=word_dict,
                                           num_topics=num_topics)

    # Print the Keyword in the 10 topics
    doc_lda = lda_model[corpus]
    print(lda_model.print_topics())
    print(lda_model.log_perplexity(corpus))

    pass


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


def filter_questions(document_vectors, documents, n_docs, diet_v, vega_v, meat_v, cook_v, animal_v):
    filtered_docs = []
    for i in range(n_docs):
        total_score = 0
        for diet_vector in diet_v:
            total_score += get_cosine_similarity(diet_vector, document_vectors[i])
        for vega_vector in vega_v:
            total_score += get_cosine_similarity(vega_vector, document_vectors[i])
        # for meat_vector in meat_v:
        #     total_score += get_cosine_similarity(meat_vector, document_vectors[i])
        for cook_vector in cook_v:
            total_score += -get_cosine_similarity(cook_vector, document_vectors[i])
        for animal_vector in animal_v:
            total_score += -get_cosine_similarity(animal_vector, document_vectors[i])
        if total_score > 0:
            filtered_docs.append(documents[i])
            print(total_score)

    print(len(filtered_docs))
    return filtered_docs


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


if __name__ == '__main__':
    question_text = load_data('data/questions.p')
    answer_text = load_data('data/answers.p', type='answer')
    answer_text = clean_answers(answer_text)
    #tokenized_answer_text = tokenize_text(answer_text['Answer'])
    #processed_answer_text = remove_stopwords(tokenized_answer_text)
    data_with_sentiment = estimate_sentiment(answer_text)

    # tokenized_text = tokenize_text(question_text)
    # processed_text = remove_stopwords(tokenized_text)
    # n_docs = len(processed_text)
    # # words_dict, text_corpus = create_input_lda(processed_text)
    # document_vecs = create_document_vec(tokenized_text)
    # # apply_lda(words_dict, text_corpus, 10)
    # dw, vw, mw, cw, aw = create_topic_vectors()
    # filter_questions(document_vecs, tokenized_text, n_docs, dw, vw, mw, cw, aw)

    # lda_model = LatentDirichletAllocation(n_topics=3,  # Number of topics
    #                                       max_iter=20,  # Max learning iterations
    #                                       learning_method='online',
    #                                       random_state=100,  # Random state
    #                                       batch_size=128,  # n docs in each learning iter
    #                                       evaluate_every=-1,  # compute perplexity every n iters, default: Don't
    #                                       n_jobs=-1,  # Use all available CPUs
    #                                       )
    # lda_output = lda_model.fit_transform(text_corpus)
    #
    # print(lda_model)  # Model attributes
