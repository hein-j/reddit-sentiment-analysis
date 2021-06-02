from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import pandas as pd


def analyze(cleaned_output):
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

    return df
