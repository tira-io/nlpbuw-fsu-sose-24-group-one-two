from pathlib import Path

from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import re
if __name__ == "__main__":
    def Convert_lower_remove_punctuation(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def Suffix_Removal(word):
        if word.endswith("ing"):
            return word[:-3]
        else:#comment
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

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", f"authorship-verification-validation-20240408-training"
    )

    # Load the model and make predictions
    model = load(Path(__file__).parent / "model.joblib")
    predictions = model.predict(df["text"].apply(preprocess))
    #print(predictions)
    df["generated"] = predictions
    df = df[["id", "generated"]]
    print(df)

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
