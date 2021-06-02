import emoji
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import emoji
import en_core_web_sm


def preprocess(relevant_strings):
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

    return lemmatized_tokens
