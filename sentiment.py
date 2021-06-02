import praw
import sys
from modules.parse import parse
from modules.gather import gather
from modules.preprocess import preprocess
from modules.analyze import analyze
from modules.visualize import visualize

# parse args
args = parse()
subreddit_str = args.subreddit
key_phrase = getattr(args, 'key phrase')
show_neutral = args.show_neutral

# establish a reddit instance with praw
print('establishing reddit instance...')

try:
    reddit = praw.Reddit("bot")
except:
    sys.exit('ERROR: Failed to establish a reddit instance. Have you correctly set up your praw.ini file? See README.md for more detail.')

print('connecting to subreddit...')
subreddit = reddit.subreddit(subreddit_str)

# Get user inputs to analyze
relevant_strings = gather(subreddit, key_phrase)

# preprocess the data
cleaned_output = preprocess(relevant_strings)

# apply sentiment analysis
df = analyze(cleaned_output)

# remove neutral words
if not show_neutral:
    print('removing neutral words...')
    df = df.loc[df['label'] != 0]
    if len(df.index) == 0:
        sys.exit('ERROR: No words found with positive or negative associations.')

visualize(df)
