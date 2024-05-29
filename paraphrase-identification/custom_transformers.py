import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import spacy

nlp = spacy.load('en_core_web_md')

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
punctuation_table = str.maketrans('', '', string.punctuation)

def preprocess(text):
    # Tokenize
    words = word_tokenize(text)
    # Lowercase
    words = [word.lower() for word in words]
    # Remove punctuation and stopwords
    words = [word.translate(punctuation_table) for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

class TfidfEmbeddingVectorizer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    def fit(self, X, y=None):
        combined_text = X['sentence1'] + " " + X['sentence2']
        self.vectorizer.fit(combined_text)
        return self
    
    def transform(self, X):
        tfidf_matrix1 = self.vectorizer.transform(X['sentence1'])
        tfidf_matrix2 = self.vectorizer.transform(X['sentence2'])
        cosine_similarities = cosine_similarity(tfidf_matrix1, tfidf_matrix2).diagonal()
        return cosine_similarities.reshape(-1, 1)

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
