import argparse


def parse():
    print('parsing arguments and options...')

    parser = argparse.ArgumentParser(description="Get the sentiment of a subreddit on a key word/phrase")
    parser.add_argument('subreddit', type=str, help='name of subreddit')
    parser.add_argument('key', type=str, help='word or phrase you want to run by the subreddit')
    parser.add_argument('--show-neutral', '-n', help='include neutral words in barplot', action='store_true')

    return parser.parse_args()