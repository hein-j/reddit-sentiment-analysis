import praw
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import emoji
import en_core_web_sm
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import pandas as pd
import plotly.express as px
from nltk import FreqDist


# TODO add progress notes
# TODO vat args

# TODO handle error when they return nothing
subreddit_str = 'popular'
key_phrase = 'book'

# Establish a reddit instance with praw
# Set up a praw.ini file in your project directory with client_id, client_secret, and user_agent.
# See https://praw.readthedocs.io/en/latest/getting_started/configuration/prawini.html
reddit = praw.Reddit("bot1")

subreddit = reddit.subreddit(subreddit_str)

# TODO unlimit
# Get user inputs to analyze
submissions = subreddit.search(key_phrase, limit=5)

relevant_strings = []

for submission in submissions:
    relevant_strings.append(str(submission.title))
    if submission.selftext:
        relevant_strings.append(str(submission.selftext))
    for comment in submission.comments.list():
        if isinstance(comment, praw.models.MoreComments):
            continue
        relevant_strings.append(str(comment.body))

# Preprocess the inputs
string_uncleaned = ','.join(relevant_strings)
# remove emojis
string_emojiless = emoji.get_emoji_regexp().sub(u'', string_uncleaned)

# tokenize & clean string
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|http\S+')
tokenized_string = tokenizer.tokenize(string_emojiless)

# lower case the string
lower_string_tokenized = [word.lower() for word in tokenized_string]

# TODO remove like

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

counts = df.label.value_counts(normalize=True) * 100

fig = px.bar(x=counts.index,
             y=counts,
             title="Percentage of Words by Sentiment",
             labels={
                 'y': '%',
                 'x': 'Sentiment'
             })

fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=[-1, 0, 1],
        ticktext=['Negative', 'Neutral', 'Positive']
    )
)

fig.show()

# further visualization

positive_words = list(df.loc[df['label'] == 1].word)

positive_frequency = FreqDist(positive_words)
pos_freq = positive_frequency.most_common(20)

negative_words = list(df.loc[df['label'] == -1].word)
negative_frequency = FreqDist(negative_words)
neg_freq = negative_frequency.most_common(20)

# visualize top words
Pos_words = [str(p) for p in pos_freq]
Pos_words_string = ','.join(Pos_words)

Neg_words = [str(n) for n in neg_freq]
Neg_words_string = ','.join(Neg_words)

pos_freq_df = pd.DataFrame(pos_freq)
pos_freq_df = pos_freq_df.rename(columns={0: 'Bar Graph of Frequent Words', 1: 'Count'}, inplace=False)

fig = px.bar(pos_freq_df, x='Bar Graph of Frequent Words', y='Count', title='Commonly Used Positive Words By Count')
fig.show()

neg_freq_df = pd.DataFrame(neg_freq)
neg_freq_df = neg_freq_df.rename(columns={0: 'Bar Graph of Frequent Words', 1: 'Count'}, inplace=False)

fig = px.bar(neg_freq_df, x='Bar Graph of Frequent Words', y='Count', title='Commonly Used Negative Words By Count')
fig.show()


