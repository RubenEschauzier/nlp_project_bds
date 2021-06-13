import pandas as pd
from answer_filter import tokenize_text, remove_stopwords, lemmatize_text, load_data, clean_answers, get_time
from LDA import LDA_scikit_gridsearch, divide_documents, statistics_categories_LDA, plot_grid_search
from predefined_topic import create_document_vec, create_topic_vectors, categorise_answers, categorise_questions, \
    statistics_categories_manual
from sentiment_mining import estimate_sentiment, get_average_sentiment, estimate_subjectivity, \
    get_sentiment_per_question


def main_sentiment():
    answers = load_data('data/answers.p', type_data='answer')
    answers = clean_answers(answers)
    sent_answers = estimate_sentiment(answers)
    pd.DataFrame.hist(sent_answers['binary_score'])
    return sent_answers


def main_categorise(clean_text, text_df, lemma_answers, raw_answers):
    print('{}: Creating document vectors'.format(get_time()))
    document_vecs = create_document_vec(clean_text)

    print('{}: Creating topic vectors'.format(get_time()))
    dw, vw, mw, cw, aw = create_topic_vectors()
    n_docs = len(clean_text)

    # filtered_questions = filter_questions(document_vecs, clean_text, n_docs, dw, vw, mw, cw, aw)
    print('{}: Categorising answers using manual method'.format(get_time()))
    d_doc, v_doc, m_doc, c_doc, a_doc, answer_df = categorise_answers(document_vecs, clean_text, n_docs, dw, vw, mw, cw,
                                                                      aw, text_df)

    print('{}: Training LDA and dividing the answers'.format(get_time()))
    answer_df = divide_documents(lemma_answers, 6, 0.1, 0.1, answer_df)

    # print('{}: Calculating statistics, this takes a while'.format(get_time()))
    # statistics_categories_manual(d_doc, v_doc, m_doc, c_doc, a_doc, question_df, raw_answers)
    statistics_categories_LDA(answer_df)
    # LDA_scikit(lemma_answers)
    LDA_results = pd.read_csv('data/topics')

    # question_df = estimate_sentiment(question_df)
    # get_average_sentiment(question_df)
    return answer_df


def main_question_filter(type_text):
    global question_df
    grid_search_lda = False
    apply_lda = True
    apply_predefined = True
    apply_predefined_questions = True
    apply_sentiment = True
    get_statistics_category = True

    assert type_text == 'Question' or type_text == 'Answer'
    print('{}: Loading data'.format(get_time()))
    question_df = load_data('data/questions.p', type_data='question').reset_index(drop=False)
    num_questions = question_df.shape[0]
    question_text = question_df['Question'].values

    answer_df = load_data('data/answers.p', type_data='answer')
    answer_df = clean_answers(answer_df)
    answer_text = answer_df['Answer'].values

    # Determine the number of answers per question and add to question df
    num_answers = []
    for i in range(num_questions):
        num_answers.append(answer_df[answer_df['QuestionID'] == i].shape[0])

    question_df['n_answers'] = num_answers

    tokenized_text = tokenize_text(question_text)
    processed_text = remove_stopwords(tokenized_text)
    lemmatized_text = lemmatize_text(processed_text)

    document_vecs = create_document_vec(lemmatized_text)

    print('{}: Creating topic vectors'.format(get_time()))
    dw, vw, mw, cw, aw = create_topic_vectors()
    n_docs = len(lemmatized_text)

    print('{}: Categorising questions into vegetarian or meat'.format(get_time()))
    question_df = categorise_questions(vw, mw, document_vecs, question_df, n_docs)

    return question_df, answer_df

    if grid_search_lda:
        print('{}: Applying gridsearch for LDA'.format(get_time()))
        LDA_scikit_gridsearch(lemmatized_text)

    if apply_lda:
        question_df = divide_documents(lemmatized_text, 6, 0.1, 0.1, question_df)

    if get_statistics_category:
        if apply_lda:
            statistics_categories_LDA(question_df, 'LDA_topics', type_text, num_topics=6)
        if apply_predefined:
            statistics_categories_LDA(question_df, 'category_w2v', type_text, num_topics=5)

    if apply_sentiment:
        print('\n {}: Estimating sentiment \n'.format(get_time()))
        question_df = estimate_sentiment(question_df, type_text)
        question_df = estimate_subjectivity(question_df, type_text)
        if apply_lda and apply_predefined:
            get_average_sentiment(question_df)

    print(question_df)


def main_answer_processor(question_df, answer_df, grid_search = False):
    category_answers = []
    questions_0 = question_df[question_df['category_w2v'] == 0]['index'].values
    questions_1 = question_df[question_df['category_w2v'] == 1]['index'].values
    for i in range(answer_df.shape[0]):
        if answer_df['QuestionID'].iloc[i] in questions_0:
            category_answers.append(0)
        if answer_df['QuestionID'].iloc[i] in questions_1:
            category_answers.append(1)
    answer_df['question_category'] = category_answers
    answer_df = answer_df.sort_values('question_category')

    topics_lda = []
    num_topics = [4, 2]

    for i in range(2):
        print('{}: Processing text'.format(get_time()))
        subset_df = answer_df[answer_df['question_category'] == i]
        answer_text = subset_df['Answer'].values

        tokenized_text = tokenize_text(answer_text)
        processed_text = remove_stopwords(tokenized_text)
        lemmatized_text = lemmatize_text(processed_text)
        if grid_search:
            print('{}: Applying gridsearch for LDA'.format(get_time()))
            search_results = LDA_scikit_gridsearch(lemmatized_text)
            plot_grid_search(search_results.cv_results_, [2, 4, 6, 8, 10, 12, 14])

        dominant_topics = divide_documents(lemmatized_text, num_topics[i], 0, 0, answer_df)
        topics_lda.extend(dominant_topics)

    answer_df['category_lda'] = topics_lda
    print('{}: Calculating statistics LDA topic allocation'.format(get_time()))
    for j in range(2):
        subset_df = answer_df[answer_df['question_category'] == j]
        statistics_categories_LDA(subset_df, 'category_lda', 'Answer')

    print('\n {}: Estimating sentiment \n'.format(get_time()))
    answer_df = estimate_sentiment(answer_df, 'Answer')
    answer_df = estimate_subjectivity(answer_df, 'Answer')
    answer_df.to_csv('full_result_df')
    question_df.to_csv('full_question_df')
    get_average_sentiment(answer_df, num_topics)

def main_summary():
    answer_df = pd.read_csv('data/full_result_df')
    question_df = pd.read_csv('data/full_question_df')
    question_df = estimate_subjectivity(question_df, 'Question')

    print('{}: Getting statistics for the LDA made categories'.format(get_time()))
    for j in range(2):
        subset_df = answer_df[answer_df['question_category'] == j]
        statistics_categories_LDA(subset_df, 'category_lda', 'Answer')

    print('{} Getting categories of the division of answers to questions based on meat vs vega in questions')
    statistics_categories_manual(answer_df, 'Answer')
    get_average_sentiment(answer_df, [4, 2])
    question_df = get_sentiment_per_question(question_df, answer_df)
    print(question_df)



if __name__ == '__main__':
    #q_df, a_df = main_question_filter('Question')
    #main_answer_processor(q_df, a_df, grid_search=False)
    main_summary()
