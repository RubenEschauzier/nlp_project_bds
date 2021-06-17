#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 18:19:35 2021

@author: stanthijssen
"""
import pandas as pd
import numpy as np
from answer_filter import tokenize_text, remove_stopwords, lemmatize_text, load_data, clean_answers, get_time, get_cosine_similarity
from LDA import LDA_scikit_gridsearch, divide_documents, statistics_categories_LDA, plot_grid_search
from create_conflict_score import conflict_score
from predefined_topic import create_document_vec, create_topic_vectors, \
    statistics_categories_manual
from sentiment_mining import estimate_sentiment, get_average_sentiment, estimate_subjectivity, \
    get_sentiment_per_question
import itertools
import operator
import string
import gensim
from tqdm import tqdm
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from datetime import datetime
from collections import Counter
import time
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pdflatex as pdflatex
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use("pgf")
import os
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
import matplotlib.pylab as pylab
import pandas as pd 
import numpy as np
import time
from wordcloud import WordCloud
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

plt.style.use('seaborn-whitegrid')
params = {'legend.fontsize': 18,
          'figure.figsize': (8, 6),
          'axes.labelsize': 18,
          'axes.titlesize': 18,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'xtick.minor.visible': True, 
          'axes.grid': True,
          'axes.linewidth': 1.0,
          'axes.titlecolor': 'black',
          'axes.labelcolor': 'black',
          'lines.markersize': 8,
          'lines.linewidth': 0.8,
          'pgf.texsystem': 'pdflatex',
          'errorbar.capsize': 8,
          'font.family': 'serif',
          'text.usetex': True,
          'pgf.rcfonts': False,
          'figure.autolayout': True}
pylab.rcParams.update(params)
matplotlib.rc('axes',edgecolor='lightgray')

def create_document_vec(input_text):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(input_text)]
    model = Doc2Vec(documents, vector_size=300, window=8, min_count=2)
    return model

def categorise_df(vega_v, meat_v, document_vectors, df, n_answers, dEps = 0):
    
    df['category_w2v'] = np.nan
    categories = []
    num_vega = 0
    num_meat = 0
    
    for i in range(n_answers):
        scores = {'vega_score': 0, 'meat_score': 0}
        
        for vega_vector in vega_v:
            scores['vega_score'] += get_cosine_similarity(vega_vector, document_vectors[i]) / len(vega_v)
        for meat_vector in meat_v:
            scores['meat_score'] += get_cosine_similarity(meat_vector, document_vectors[i]) / len(meat_v)

        #max_score = max(scores.items(), key=operator.itemgetter(1))[0]
        
        #if max_score == 'vega_score':
        #    categories.append(0)
        #    num_vega += 1
        #if max_score == 'meat_score':
        #    categories.append(1)
        #    num_meat += 1
        if scores['vega_score'] - scores['meat_score'] > dEps:
            df['category_w2v'].iloc[i] = 0
            num_vega += 1
        if scores['meat_score'] - scores['vega_score'] > dEps:
            df['category_w2v'].iloc[i] = 1
            num_meat += 1
            
    #df['category_w2v'] = categories
    print('{}: Number in vega: {}, number in meat: {}'.format(get_time(), num_vega, num_meat))

    return [num_vega, num_meat, df]

### Preparation
answer_df = pd.read_csv('data/full_result_df', index_col=0)
question_df = pd.read_csv('data/full_question_df', index_col=0)
question_df['category_w2v'] = np.nan
answer_text = answer_df['Answer'].values
tokenized_text = tokenize_text(answer_text)
processed_text = remove_stopwords(tokenized_text)
lemmatized_text = lemmatize_text(processed_text)
print('{}: Creating topic vectors'.format(get_time()))
dw, vw, mw, cw, aw = create_topic_vectors()
n_docs = len(lemmatized_text)
document_vecs = create_document_vec(lemmatized_text)


### Table mean sentiment
print('{}: Categorising questions into vegetarian or meat'.format(get_time()))
mFriction = np.nan * np.zeros((6,7))
vEps = np.linspace(0,0.05,6)

for i in range(len(vEps)):
    dEps = vEps[i]
    print('Epsilon ' + str(dEps))
    num_vega, num_meat, answer_df_temp = categorise_df(vw, mw, document_vecs, answer_df, n_docs, dEps)
    mFriction[i,:3] = [dEps, num_vega, num_meat]
    dfTemp = answer_df_temp.loc[(answer_df['question_category'] == 0) & (answer_df['category_w2v'] == 0), :]
    mFriction[i, 3] = np.mean(dfTemp['score'])
    dfTemp = answer_df_temp.loc[(answer_df['question_category'] == 0) & (answer_df['category_w2v'] == 1), :]
    mFriction[i, 4] = np.mean(dfTemp['score'])
    dfTemp = answer_df_temp.loc[(answer_df['question_category'] == 1) & (answer_df['category_w2v'] == 0), :]
    mFriction[i, 5] = np.mean(dfTemp['score'])
    dfTemp = answer_df_temp.loc[(answer_df['question_category'] == 1) & (answer_df['category_w2v'] == 1), :]
    mFriction[i, 6] = np.mean(dfTemp['score'])
    print('{}'.format(get_time()))
    print('============================================================================')

print(mFriction)
dfFriction=pd.DataFrame(mFriction)
print(dfFriction.to_latex(float_format="{:0.4f}".format ))

### Table std sentiment
mFrictionStd = np.nan * np.zeros((6,7))
for i in range(len(vEps)):
    dEps = vEps[i]
    print('Epsilon ' + str(dEps))
    num_vega, num_meat, answer_df_temp = categorise_df(vw, mw, document_vecs, answer_df, n_docs, dEps)
    mFrictionStd[i,:3] = [dEps, num_vega, num_meat]
    dfTemp = answer_df_temp.loc[(answer_df['question_category'] == 0) & (answer_df['category_w2v'] == 0), :]
    mFrictionStd[i, 3] = np.std(dfTemp['score'])
    dfTemp = answer_df_temp.loc[(answer_df['question_category'] == 0) & (answer_df['category_w2v'] == 1), :]
    mFrictionStd[i, 4] = np.std(dfTemp['score'])
    dfTemp = answer_df_temp.loc[(answer_df['question_category'] == 1) & (answer_df['category_w2v'] == 0), :]
    mFrictionStd[i, 5] = np.std(dfTemp['score'])
    dfTemp = answer_df_temp.loc[(answer_df['question_category'] == 1) & (answer_df['category_w2v'] == 1), :]
    mFrictionStd[i, 6] = np.std(dfTemp['score'])
    print('{}'.format(get_time()))
    print('============================================================================')

print(mFrictionStd)
dfFrictionStd=pd.DataFrame(mFrictionStd)
print(dfFrictionStd.to_latex(float_format="{:0.4f}".format ))

### sentiment vs subjectivity - Regression still needs to be done
# answer_df['AnswerID'] = np.linspace(1, 108812, 108812)
# question_df = estimate_subjectivity(question_df, 'Question')
# question_df = get_sentiment_per_question(question_df, answer_df)
# model = smf.ols(formula = 'std_sentiment ~ score_subjectivity', data= question_df)
#  result = model.fit()
# print(result.summary())

### Rsults section 2.3
## Questions
question_df = pd.read_csv('data/full_question_df', index_col=0)
t = time.time()
s0 = question_df['Question'].str.split(expand=True).stack().value_counts()
print(s0[0:10])
print(np.sum(s0))
print(time.time() - t)

# Eliminate stopwords, ignother phases as won't impact first 20 results
dfS0 = s0.to_frame()
for l in stopwords:
    try: 
        dfS0 = dfS0.drop(l)
    except:
        dfS0 = dfS0

s0 = dfS0.squeeze()

# Histogram [Note: divide  by 100 before]
plt.figure()
plot = s0[0:20].sort_values(ascending=False)[0:20].plot.bar()
fig = plot.get_figure()
plt.yticks(rotation=90)
#fig.savefig('results/histogramQ.png')
plt.savefig('results/histogramQ.pdf', backend = 'pgf')

# Wordcloud
questions = question_df['Question'].str.split(expand=True)
wordcloud = WordCloud().generate_from_frequencies(frequencies=s0)
wordcloud.to_file("results/wordcloudQ.eps")
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

###Answers
answer_df = pd.read_csv('data/full_result_df', index_col=0)
s0A = answer_df['Answer'].str.split(expand=True).stack().value_counts()

# Histogram [Note: divide  by 10,000 before]
plt.figure(figsize = (6,6))
plot = s0A[0:20].sort_values(ascending=False)[0:20].plot.bar()
plt.yticks(rotation=90)
fig = plot.get_figure()
plt.savefig('results/histogramA.pdf', backend = 'pgf')

# Wordcloud
wordcloud = WordCloud().generate_from_frequencies(frequencies=s0A)
wordcloud.to_file("wordcloudA.eps")
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
