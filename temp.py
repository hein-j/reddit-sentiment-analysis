import praw
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import emoji
import en_core_web_sm
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import pandas as pd
import plotly.express as px
from nltk import FreqDist
import sys
import argparse

print('parsing arguments and options...')

parser = argparse.ArgumentParser(description="Get the sentiment of a subreddit on a key phrase")
parser.add_argument('subreddit', type=str, help='name of subreddit')
parser.add_argument('key phrase', type=str, help='word or phrase you want to run by the subreddit')
parser.add_argument('--show-neutral', '-n', help='include neutral words in barplot', action='store_true')
args = parser.parse_args()
subreddit_str = args.subreddit
key_phrase = getattr(args, 'key phrase')
show_neutral = args.show_neutral

print('establishing reddit instance...')
# Establish a reddit instance with praw
# Set up a praw.ini file in your project directory with client_id, client_secret, and user_agent.
# See https://praw.readthedocs.io/en/latest/getting_started/configuration/prawini.html
reddit = praw.Reddit("bot1")
print('connecting to subreddit...')
subreddit = reddit.subreddit(subreddit_str)

# Get user inputs to analyze
print('searching subreddit for key phrase...')
submissions = subreddit.search(key_phrase, limit=5)
relevant_strings = []
print('gathering texts for analysis...')
try:
    for submission in submissions:
        if submission.selftext:
            relevant_strings.append(str(submission.selftext))
        for comment in submission.comments.list():
            if isinstance(comment, praw.models.MoreComments):
                continue
            relevant_strings.append(str(comment.body))
    if len(relevant_strings) == 0:
        raise Exception
except:
    sys.exit('ERROR: No posts were found for the provided subreddit and key phrase.')

print('preprocessing the data...')
# Preprocess the inputs
string_uncleaned = ','.join(relevant_strings)
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
tokens_without_sw = [word for word in text if word != "like" and word not in all_stopwords]

# normalize via lemmatizing
lemmatizer = WordNetLemmatizer()

lemmatized_tokens = ([lemmatizer.lemmatize(w) for w in tokens_without_sw])

cleaned_output = lemmatized_tokens

# Apply sentiment analyzer using VADER
print('applying sentiment analysis...')
sia = SIA()
results = []

for word in cleaned_output:
    pol_score = sia.polarity_scores(word)
    pol_score['word'] = word
    results.append(pol_score)
pd.set_option('display.max_columns', None, 'max_colwidth', None)
df = pd.DataFrame.from_records(results)

# add a label
df['label'] = 0
df.loc[df['compound'] > 0.10, 'label'] = 1
df.loc[df['compound'] < -0.10, 'label'] = -1

# remove neutral words
if not show_neutral:
    print('removing neutral words...')
    df = df.loc[df['label'] != 0]
    if len(df.index) == 0:
        sys.exit('ERROR: No words found with positive or negative associations.')

counts = df.label.value_counts(normalize=True) * 100

print('generating visual representation...')

main_fig = px.bar(x=counts.index,
             y=counts,
             title="Percentage of Words by Sentiment",
             labels={
                 'y': '%',
                 'x': 'Sentiment'
             })

main_fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=[-1, 1],
        ticktext=['Negative', 'Positive']
    )
)

# visualize top words

positive_words = list(df.loc[df['label'] == 1].word)

positive_frequency = FreqDist(positive_words)
pos_freq = positive_frequency.most_common(20)

negative_words = list(df.loc[df['label'] == -1].word)
negative_frequency = FreqDist(negative_words)
neg_freq = negative_frequency.most_common(20)

pos_freq_df = pd.DataFrame(pos_freq)
pos_freq_df = pos_freq_df.rename(columns={0: 'Bar Graph of Frequent Words', 1: 'Count'}, inplace=False)

pos_fig = px.bar(pos_freq_df, x='Bar Graph of Frequent Words', y='Count', title='Commonly Used Positive Words By Count')
pos_fig.show()

neg_freq_df = pd.DataFrame(neg_freq)
neg_freq_df = neg_freq_df.rename(columns={0: 'Bar Graph of Frequent Words', 1: 'Count'}, inplace=False)

neg_fig = px.bar(neg_freq_df, x='Bar Graph of Frequent Words', y='Count', title='Commonly Used Negative Words By Count')
neg_fig.show()

main_fig.show()

