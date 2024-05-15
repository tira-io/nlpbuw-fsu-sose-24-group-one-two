import re

from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from joblib import dump
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

if __name__ == "__main__":

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    merged_data = pd.merge(text_validation, targets_validation, on='id')
    lang_ids = [
        "af",
        "az",
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "hr",
        "it",
        "ko",
        "nl",
        "no",
        "pl",
        "ru",
        "ur",
        "zh",
    ]
    stopwords = {
        lang_id: set(
            (Path(__file__).parent / "stopwords" / f"stopwords-{lang_id}.txt")
            .read_text()
            .splitlines()
        )
        - set(("(", ")", "*", "|", "+", "?"))  # remove regex special characters
        for lang_id in lang_ids
    }
    lt=WordNetLemmatizer()
    st=PorterStemmer()
    def First_process(input_txt):
        input_txt = re.sub(r'[^a-zA-Z\s]', '', input_txt)
        input_txt = word_tokenize(input_txt)
        input_txt = [lt.lemmatize(token) for token in input_txt]
        input_txt = [st.stem(token) for token in input_txt]
        input_txt = [word for word in input_txt if word not in stopwords]
        input_txt = ' '.join(input_txt)
        return input_txt
    merged_data["text"]=merged_data["text"].apply(First_process)
    TFV = TfidfVectorizer(max_features=9990)
    X = TFV.fit_transform(merged_data["text"])
    encod = LabelEncoder()
    Y = encod.fit_transform(merged_data["lang"])
    last = pd.DataFrame(np.c_[merged_data["text"], Y], columns=["text", "lang"])
    model = Pipeline([
        ("vectorizer", TFV),
        ("classifier", SVC(kernel='linear'))  # Using linear kernel for SVM
    ])
    model.fit(merged_data["text"], merged_data["lang"])

    # Save the model
    dump(model, Path(__file__).parent / "model.joblib")
    model = load(Path(__file__).parent / "model.joblib")
    predictions = model.predict(text_validation["text"])
    text_validation["lang"] = predictions
    df = text_validation[["id", "lang"]]

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
   