import string
import os
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TAGALOG_STOPWORDS_PATH = os.path.join(BASE_DIR, "stopwords", "tagalog_stopwords.txt")

# Load stopwords once
with open(TAGALOG_STOPWORDS_PATH, "r", encoding="utf-8") as f:
    tagalog_stop = set([line.strip() for line in f if line.strip()])
english_stop = set(stopwords.words("english"))
ALL_STOPWORDS = english_stop.union(tagalog_stop)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._clean_text(text) for text in X]

    def _clean_text(self, mess):
        # Remove punctuation
        nopunc = [char for char in mess if char not in string.punctuation]
        nopunc = ''.join(nopunc)
        # Remove stopwords
        clean_words = [word for word in nopunc.split() if word.lower() not in ALL_STOPWORDS]
        return " ".join(clean_words)
