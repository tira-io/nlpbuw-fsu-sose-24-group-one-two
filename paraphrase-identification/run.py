from pathlib import Path
from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from custom_transformers import preprocess, TfidfEmbeddingVectorizer, NGramFeatures, SemanticSimilarity
import nltk
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

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    ).set_index("id")
    df['sentence1'] = df['sentence1'].apply(preprocess)
    df['sentence2'] = df['sentence2'].apply(preprocess)


    # Predict using the model stored
    model = load(Path(__file__).parent / "model.joblib")
    predictions = model.predict(df[['sentence1', 'sentence2']])

    # Add predictions to dataframe
    df["label"] = predictions
    print(df)
    df = df.drop(columns=["sentence1", "sentence2"]).reset_index()
    print(df)

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
