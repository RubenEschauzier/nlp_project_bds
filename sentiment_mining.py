import pandas as pd
import numpy as np
import nltk
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from answer_filter import get_time


def estimate_sentiment(text_data, col_name) -> pd.DataFrame:
    """

    :param text_data: Dataframe containing the answers we want to estimate sentiment of
    :return: dataframe with both a continous and binary score of sentiment
    """
    total_scores = []
    binary_score = []
    for answer in tqdm(text_data[col_name]):
        num_sent = 0
        answer_score = 0
        sent_answer = nltk.sent_tokenize(answer)
        for i, sentence in enumerate(sent_answer):
            analyser = SentimentIntensityAnalyzer()
            score = analyser.polarity_scores(sentence)
            answer_score += score['compound']
            num_sent += 1
        total_scores.append(answer_score/num_sent)
        if answer_score > 0.05:
            binary_score.append(1)
        elif answer_score < -0.05:
            binary_score.append(-1)
        else:
            binary_score.append(0)
    text_data['binary_score'] = binary_score
    text_data['score'] = total_scores
    return text_data

def estimate_subjectivity(text_data, col_name):
    total_scores = []
    for answer in tqdm(text_data[col_name]):
        blob = TextBlob(answer)
        total_scores.append(blob.sentiment.subjectivity)
    text_data['score_subjectivity'] = total_scores
    return text_data

def get_sentiment_per_question(question_df, answer_df):

    mean_sent_list = []
    std_sent_list = []
    for i in range(question_df.shape[0]):
        answer_subset = answer_df[answer_df['QuestionID'] == i]
        mean_sent_list.append(answer_subset['score'].mean())
        std_sent_list.append(answer_subset['score'].std())
    question_df['mean_sentiment'] = mean_sent_list
    question_df['std_sentiment'] = std_sent_list

    return question_df


def get_sentiment_per_answer(answer_df):

    mean_sent_list = []
    std_sent_list = []
    for i in range(answer_df.shape[0]):
        #print(answer)
        if i % 1000 == 0:
            print('Iteration :' + str(i) + ' of 108812({})'.format(get_time()))
        answer_subset = answer_df[answer_df['AnswerID'] == i]
        mean_sent_list.append(answer_subset['score'].mean())
        std_sent_list.append(answer_subset['score'].std())
    answer_df['mean_sentiment'] = mean_sent_list
    answer_df['std_sentiment'] = std_sent_list

    return answer_df

def get_average_sentiment(answer_df, num_topics):
    print('Question topic 0 = vega, question topic 1 = meat')
    for x in range(2):
        subset_df = answer_df[answer_df['question_category'] == x]
        for i in range(num_topics[x]):
            sentiment = subset_df['score'][subset_df['category_lda'] == i]
            print('{} For Question topic ({}) and LDA topic {} average sentiment is equal to: {}'.
                  format(get_time(), x, i, sentiment.mean()))

def evaluate_performance(annotated_dataframe):
    print(np.unique(annotated_dataframe['Ground_Truth'].values, return_counts=True))
    estimated_score = []
    for score in annotated_dataframe['score']:
        if score > 0.05:
            estimated_score.append(1)
        elif score < -0.05:
            estimated_score.append(-1)
        else:
            estimated_score.append(0)
    print(np.unique(np.array(estimated_score), return_counts=True))
    result = confusion_matrix(annotated_dataframe['Ground_Truth'], estimated_score, labels = [-1, 0, 1])

    print(result)

