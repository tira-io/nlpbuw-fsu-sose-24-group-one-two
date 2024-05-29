from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy
from scipy.sparse import hstack
from transformers import BertTokenizer, BertModel
# Load spaCy model
nlp = spacy.load('en_core_web_md')

# Custom transformer for computing n-gram features
class NGramFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, ngram_range=(1, 3)):
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, analyzer='word')

    def fit(self, X, y=None):
        texts1 = X['sentence1'].tolist()
        texts2 = X['sentence2'].tolist()
        self.vectorizer.fit(texts1 + texts2)
        return self

    def transform(self, X):
        texts1 = X['sentence1'].tolist()
        texts2 = X['sentence2'].tolist()
        ngram_matrix1 = self.vectorizer.transform(texts1)
        ngram_matrix2 = self.vectorizer.transform(texts2)
        return hstack([ngram_matrix1, ngram_matrix2])

# Custom transformer for computing semantic similarity using spaCy
class SemanticSimilarity(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts1 = X['sentence1'].tolist()
        texts2 = X['sentence2'].tolist()
        similarities = []
        for doc1, doc2 in zip(texts1, texts2):
            similarity = nlp(doc1).similarity(nlp(doc2))
            similarities.append([similarity])
        return np.array(similarities)
