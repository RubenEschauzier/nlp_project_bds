import pandas as pd
import nltk
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
        answer_score = 0
        sent_answer = nltk.sent_tokenize(answer)
        for sentence in sent_answer:
            analyser = SentimentIntensityAnalyzer()
            score = analyser.polarity_scores(sentence)
            answer_score += score['compound']
        total_scores.append(answer_score)
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


def get_average_sentiment(answer_df):
    for i in range(1, 6):
        sentiment = answer_df['score'][answer_df['category_w2v'] == i]
        print('{} For Topic (manual) {} average sentiment is equal to: {}'.format(get_time(), i, sentiment.mean()))

    for j in range(6):
        sentiment = answer_df['score'][answer_df['LDA_topics'] == j]
        print('{} For Topic (LDA) {} average sentiment is equal to: {}'.format(get_time(), j, sentiment.mean()))

