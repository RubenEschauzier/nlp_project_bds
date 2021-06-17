from scipy.stats import hmean
import pandas as pd


def conflict_score(question_df, answer_df):
    pd.options.mode.chained_assignment = None
    # Drop any questions with only one answer, which causes the std to be nan
    question_df = question_df.dropna()

    # Normalize sentiment between 0 and 1
    std_sentiment = question_df['std_sentiment'].values
    std_sentiment = (std_sentiment - min(std_sentiment)) / (max(std_sentiment) - min(std_sentiment))

    # Get subjectivity from df
    subjectivity = question_df['score_subjectivity'].values

    # Create data and take harmonic mean
    data = [[std, sub] for std, sub in zip(std_sentiment, subjectivity)]
    h_means = hmean(data, axis=1)

    question_df['conflict_score'] = h_means
    print(question_df.columns)
    return question_df
