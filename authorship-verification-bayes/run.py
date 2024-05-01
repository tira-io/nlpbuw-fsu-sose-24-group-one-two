from pathlib import Path

from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import re
if __name__ == "__main__":
    def preprocessingText(text):
        text = re.sub(r'[^\w\s]', '', text)
        return text
    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", f"authorship-verification-validation-20240408-training"
    )

    # Load the model and make predictions
    model = load(Path(__file__).parent / "model.joblib")
    predictions = model.predict(df["text"].apply(preprocessingText))
    #print(predictions)
    df["generated"] = predictions
    df = df[["id", "generated"]]
    print(df)

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
