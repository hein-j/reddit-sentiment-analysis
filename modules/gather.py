import praw
import sys


def gather(subreddit, key_phrase):
    print('searching subreddit for key phrase...')
    submissions = subreddit.search(key_phrase, limit=5)
    relevant_strings = []
    print('gathering texts for analysis...')
    try:
        for submission in submissions:
            print('...')
            if submission.selftext:
                relevant_strings.append(str(submission.selftext))
            for comment in submission.comments.list():
                if isinstance(comment, praw.models.MoreComments):
                    continue
                relevant_strings.append(str(comment.body))
        if len(relevant_strings) == 0:
            raise Exception
        return relevant_strings
    except:
        sys.exit('ERROR: No posts were found for the provided subreddit and key phrase.')
