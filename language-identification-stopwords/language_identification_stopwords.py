from pathlib import Path
import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from tqdm import tqdm

# Load stopwords for each language
def load_stopwords():
    stopwords = {}
    lang_ids = [
        "af", "az", "bg", "cs", "da", "de", "el", "en", "es", "fi",
        "fr", "hr", "it", "ko", "nl", "no", "pl", "ru", "ur", "zh"
    ]
    for lang_id in lang_ids:
        stopwords[lang_id] = set(
            (Path(__file__).parent / "stopwords" / f"stopwords-{lang_id}.txt").read_text().splitlines()
        )
    return stopwords

# Predict language based on stopwords
def predict_language(text, stopwords):
    lang_scores = {lang: 0 for lang in stopwords}
    for lang, lang_stopwords in stopwords.items():
        for word in text.split():
            if word.lower() in lang_stopwords:
                lang_scores[lang] += 1
    return max(lang_scores, key=lang_scores.get)

if __name__ == "__main__":
    tira = Client()

    # Load validation data
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training")
    targets_validation = tira.pd.truths("nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training")

    # Load stopwords
    stopwords = load_stopwords()

    # Predict languages
    predictions = []
    for id, text in tqdm(zip(text_validation['id'], targets_validation['id']), total=len(text_validation)):
        lang = predict_language(text, stopwords)
        predictions.append({'id': id, 'lang': lang})

    # Save predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    pd.DataFrame(predictions).to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
