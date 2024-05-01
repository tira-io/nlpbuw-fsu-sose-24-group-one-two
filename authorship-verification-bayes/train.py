from pathlib import Path
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from tira.rest_api_client import Client
import re

if __name__ == "__main__":
    def preprocessingText(text):
        # Lowercasing
        text = text.lower()
        # Removing punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text

    # Custom stemming function
    def stem_word(word):
        #  suffix stripping
        if word.endswith("ing"):
            return word[:-3]
        else:
            return word

    # Custom lemmatization function
    def lemmatize_word(word):
        # Basic lemmatization (e.g., removing 's', 'es', 'ed', 'ing')
        if word.endswith("s") or word.endswith("es"):
            return word[:-1]
        elif word.endswith("ed"):
            return word[:-2]
        else:
            return word

    #stop word removal function
    def remove_stop_words(tokens):
        stop_words = set(["a", "an", "the", "and", "or", "but", "for", "of", "in", "on", "at", "to", "from", "with"])  # Add more stop words as needed
        return [token for token in tokens if token not in stop_words]

    def tokenize_text(text):
        return text.split()

    def preprocess(text):
        text = preprocessingText(text)
        tokens = tokenize_text(text)
        stemmed_tokens = [stem_word(token) for token in tokens]
        lemmatized_tokens = [lemmatize_word(token) for token in tokens]
        tokens_without_stop_words = remove_stop_words(lemmatized_tokens)
        processed_text = ' '.join(tokens_without_stop_words)
        return processed_text

    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    text = text.set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    df = text.join(labels.set_index("id"))
    
    # Preprocess text
    df["text"] = df["text"].apply(preprocess)

    # Train the model
    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", SVC(kernel="linear"))
    ])
    model.fit(df["text"], df["generated"])

    # Save the model
    dump(model, Path(__file__).parent / "model.joblib")
