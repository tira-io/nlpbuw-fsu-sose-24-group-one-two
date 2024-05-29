from pathlib import Path
from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from custom_transformers import preprocess, TfidfEmbeddingVectorizer, NGramFeatures, SemanticSimilarity

if __name__ == "__main__":
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
