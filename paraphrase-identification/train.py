from tira.rest_api_client import Client
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, matthews_corrcoef
from custom_transformers import NGramFeatures, SemanticSimilarity
from joblib import dump
from pathlib import Path

if __name__ == "__main__":
    # Load the data
    tira = Client()
    text = tira.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")
    labels = tira.pd.truths("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")
    df = text.join(labels)

    # Create a pipeline with n-gram features and semantic similarity
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('ngram', NGramFeatures()),
            ('semantic', SemanticSimilarity())
        ])),
        ('classifier', SVC(kernel='linear', C=1.0))  # Default hyperparameters
    ])

    # Fit the model
    pipeline.fit(df[['sentence1', 'sentence2']], df['label'])

    # Save the model
    dump(pipeline, Path(__file__).parent / "model.joblib")

    # Predict and evaluate
    y_pred = pipeline.predict(df[['sentence1', 'sentence2']])
    accuracy = accuracy_score(df['label'], y_pred)
    mcc = matthews_corrcoef(df['label'], y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"MCC: {mcc}")
