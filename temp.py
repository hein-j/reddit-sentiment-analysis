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

