from pathlib import Path

from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
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
