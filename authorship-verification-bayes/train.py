from pathlib import Path
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from tira.rest_api_client import Client
import re

if __name__ == "__main__":
    def Convert_lower_remove_punctuation(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def Suffix_Removal(word):
        if word.endswith("ing"):
            return word[:-3]
        else:
            return word

    def Normalize(word):
        if word.endswith("s") or word.endswith("es"):
            return word[:-1]
        elif word.endswith("ed"):
            return word[:-2]
        else:
            return word

    def Stop_words_removal(tokens):
        stop_words = set(["a", "an", "the", "and", "or", "but", "for", "of", "in", "on", "at", "to", "from", "with"])  # Add more stop words as needed
        return [token for token in tokens if token not in stop_words]

    def tokenize_text(text):
        return text.split()

    def preprocess(text):
        text = Convert_lower_remove_punctuation(text)
        tokens = tokenize_text(text)
        stemmed_tokens = [Suffix_Removal(token) for token in tokens]
        lemmatized_tokens = [Normalize(token) for token in tokens]
        tokens_without_stop_words = Stop_words_removal(lemmatized_tokens)
        processed_text = ' '.join(tokens_without_stop_words)
        return processed_text


    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    text = text.set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    df = text.join(labels.set_index("id"))
    
    df["text"] = df["text"].apply(preprocess)

    # Train the model
    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", SVC(kernel="linear"))
    ])
    model.fit(df["text"], df["generated"])

    # Save the model
    dump(model, Path(__file__).parent / "model.joblib")
