import pandas as pd
import plotly.express as px
from nltk import FreqDist


def visualize(df):
    print('generating visual representation...')

    counts = df.label.value_counts(normalize=True) * 100

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

    pos_fig = px.bar(pos_freq_df, x='Bar Graph of Frequent Words', y='Count',
                     title='Commonly Used Positive Words By Count')
    pos_fig.show()

    neg_freq_df = pd.DataFrame(neg_freq)
    neg_freq_df = neg_freq_df.rename(columns={0: 'Bar Graph of Frequent Words', 1: 'Count'}, inplace=False)

    neg_fig = px.bar(neg_freq_df, x='Bar Graph of Frequent Words', y='Count',
                     title='Commonly Used Negative Words By Count')
    neg_fig.show()

    main_fig.show()
