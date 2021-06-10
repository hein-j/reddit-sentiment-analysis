<h1 align="center">Reddit Sentiment Analysis</h1>

![preview](https://github.com/hein-j/reddit-sentiment-analysis/blob/main/docs/screenshot.png?raw=true)

<!-- ABOUT THE PROJECT -->
## About

Ever wonder how a subreddit feels about a certain subject? This script can provide insight. Input the name of a subreddit and a key word/phrase to view bar plot representations of the sentiment analysis. 

### Built With

* [Python3](https://www.python.org/)
* [PRAW](https://praw.readthedocs.io/en/latest/#)
* [NLTK](https://www.nltk.org/)

<!-- USAGE EXAMPLES -->
## Usage

```sentiment.py [-h] [--show-neutral] subreddit key```

```
positional arguments:
  subreddit           name of subreddit
  key                 word or phrase you want to run by the subreddit

optional arguments:
  -h, --help          show this help message and exit
  --show-neutral, -n  include neutral words in barplot
```
<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
* Python3 (3.9.5)
  * See the <a href="https://www.python.org/downloads/">official webpage</a> for download information.
* Conda
    * You are welcome to use other package management systems, but I ran into a lot of trouble with pip on a Mac M1. I installed conda through <a href="https://github.com/conda-forge/miniforge">Miniforge.</a>


### Installation
1. ```git clone https://github.com/hein-j/reddit-sentiment-analysis.git```
2. ```cd``` into project root
3. Install dependencies (into a virtual environment if you wish)
    1. For conda: ```conda create --prefix ./venv --file requirements.txt``` and then ```conda activate ./venv```
    2. If you run into trouble or are not using conda, the dependencies are:
        1. praw
        2. nltk
        3. pandas
        4. emoji
        5. spacy
        6. plotly
4. Download 'wordnet' and 'vader_lexicon' from nltk (<a href="http://www.nltk.org/data.html">See instructions here</a>)  
5. Download spacy model: ```python -m spacy download en_core_web_sm```
6. Create a reddit app (<a href="https://www.geeksforgeeks.org/how-to-get-client_id-and-client_secret-for-python-reddit-api-registration/">See here for help</a>)
7. Create a ```praw.ini``` file in project root. Name the site ```bot```. Include ```client_id```, ```client_secret```, and ```user_agent```. (<a href="https://praw.readthedocs.io/en/latest/getting_started/configuration/prawini.html">See here for help</a>)
8. You should now be good to go 🤘
<!-- CONTRIBUTING -->
## Contributing

Issues & pull requests are most welcome. 

<!-- LICENSE -->
## License

Distributed under the MIT License. See <a href="https://github.com/hein-j/reddit-sentiment-analysis/blob/main/LICENSE.txt">License</a> for more information.




<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
Big thanks to <a href="https://levelup.gitconnected.com/reddit-sentiment-analysis-with-python-c13062b862f6">this article by Jason LZP</a> for getting this project started.