import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from tira.rest_api_client import Client
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from joblib import dump
from pathlib import Path
from custom_transformers import First_process
from custom_transformers import NGramFeatures, SemanticSimilarity,TfidfEmbeddingVectorizer

if __name__ == "__main__":
    # Load the data
    tira = Client()
    text = tira.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")
    labels = tira.pd.truths("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")
    df = text.join(labels)

    # Apply preprocessing to the text data
    df['sentence1'] = df['sentence1'].apply(First_process)
    df['sentence2'] = df['sentence2'].apply(First_process)

    # Create a pipeline with n-gram features and semantic similarity
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('ngram', NGramFeatures()),
            ('semantic', SemanticSimilarity()),
            ('tfidf_embedding', TfidfEmbeddingVectorizer())
        ])),
        ('classifier', SVC(kernel='linear', C=1.0))  # Default hyperparameters
    ])
    
    # Fit the model
    pipeline.fit(df[['sentence1', 'sentence2']], df['label'])

    # Save the model
    dump(pipeline, Path(__file__).parent / "model.joblib")
