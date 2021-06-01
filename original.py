import praw
import pandas as pd
from praw.models import MoreComments
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import FreqDist
import emoji
import re
import en_core_web_sm
import spacy

# TODO do not commit them secrets.. init file?

# Establish a reddit instance with praw
# Set up a praw.ini file in your project directory with client_id, client_secret, and user_agent.
# See https://praw.readthedocs.io/en/latest/getting_started/configuration/prawini.html
reddit = praw.Reddit("bot1")

# Obtain comments from a post
# TODO explore hot, new, top
# TODO or take submission ID directly from bar
# subreddit = reddit.subreddit('popular')
# for submission in subreddit.hot(limit=5):
#     print(submission.title)
#     print('Submission ID =', submission.id, '\n')

Post1 = reddit.submission(id='nnpryq')
Comments_All = []
Post1.comments.replace_more(limit=None) # remove all nestled comments (is destructive)
for comments in Post1.comments.list():
    Comments_All.append(comments.body)

# Preprocess the comments
List1 = [str(i) for i in Comments_All]
string_uncleaned = ','.join(List1)
# remove emojis
string_emojiless = emoji.get_emoji_regexp().sub(u'', string_uncleaned)

# tokenize & clean string
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|http\S+')
tokenized_string = tokenizer.tokenize(string_emojiless)

# lower case the string
lower_string_tokenized = [word.lower() for word in tokenized_string]

# remove stop words
nlp = en_core_web_sm.load()
all_stopwords = nlp.Defaults.stop_words
text = lower_string_tokenized
tokens_without_sw = [word for word in text if not word in all_stopwords]

# normalize via lemmatizing
lemmatizer = WordNetLemmatizer()

lemmatized_tokens = ([lemmatizer.lemmatize(w) for w in tokens_without_sw])

cleaned_output = lemmatized_tokens

# Apply sentiment analyzer using VADER
import nltk.sentiment.vader
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
results = []

for sentences in cleaned_output:
    pol_score = sia.polarity_scores(sentences)
    pol_score['words'] = sentences
    results.append(pol_score)
pd.set_option('display.max_columns', None, 'max_colwidth', None)
df = pd.DataFrame.from_records(results)

# add a label
df['label'] = 0
df.loc[df['compound'] > 0.10, 'label'] = 1
df.loc[df['compound'] < -0.10, 'label'] = -1
#print(df.head())

# represent results
#print(df.label.value_counts())

import seaborn as sns
import matplotlib.pyplot as plt

# create a figure and set of subplots
# returns figure and array of axes
fig, ax = plt.subplots(figsize=(8, 8))

# # produce counts as percentage
counts = df.label.value_counts(normalize=True) * 100

sns.barplot(x=counts.index, y=counts, ax=ax)

ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")

plt.show()

# remove neutral words
df_positive_negative = df.loc[df['label'] != 0]
df_positive_negative.head()

fig, ax = plt.subplots(figsize=(8, 8))

# produce counts as percentage
counts = df_positive_negative.label.value_counts(normalize=True) * 100

sns.barplot(x=counts.index, y=counts, ax=ax)

ax.set_xticklabels(['Negative', 'Positive'])
ax.set_ylabel("Percentage")

# plt.show()

# further visualization

positive_words = list(df.loc[df['label'] == 1].words)
# print(positive_words)

positive_frequency = FreqDist(positive_words)
pos_freq = positive_frequency.most_common(20)
# print(pos_freq)

negative_words = list(df.loc[df['label'] == -1].words)
negative_frequency = FreqDist(negative_words)
neg_freq = negative_frequency.most_common(20)
print(neg_freq)

# visualize top words
Pos_words = [str(p) for p in pos_freq]
Pos_words_string = ','.join(Pos_words)

Neg_words = [str(n) for n in neg_freq]
Neg_words_string = ','.join(Neg_words)

import plotly.express as px

pos_freq_df = pd.DataFrame(pos_freq)
pos_freq_df = pos_freq_df.rename(columns={0: 'Bar Graph of Frequent Words', 1: 'Count'}, inplace=False)

fig = px.bar(pos_freq_df, x='Bar Graph of Frequent Words', y='Count', title='Commonly Used Positive Words By Count')
fig.show()

neg_freq_df = pd.DataFrame(neg_freq)
neg_freq_df = neg_freq_df.rename(columns={0: 'Bar Graph of Frequent Words', 1: 'Count'}, inplace=False)

fig = px.bar(neg_freq_df, x='Bar Graph of Frequent Words', y='Count', title='Commonly Used Negative Words By Count')
fig.show()

