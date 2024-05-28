from tira.rest_api_client import Client
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, matthews_corrcoef
import pandas as pd
from joblib import dump
from pathlib import Path
from custom_transformers import NGramFeatures, SemanticSimilarity

if __name__ == "__main__":
    # Load the data
    tira = Client()
    text = tira.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")
    labels = tira.pd.truths("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")
    df = text.join(labels)

    # Create a pipeline with n-gram features and semantic similarity
    model = Pipeline([
        ('features', FeatureUnion([
            ('ngram', NGramFeatures()),
            ('semantic', SemanticSimilarity())
        ])),
        ('classifier', SVC(kernel='linear'))  # Using linear kernel for SVM
    ])

    # Fit the model
    model.fit(df[['sentence1', 'sentence2']], df['label'])

    # Save the model
    dump(model, Path(__file__).parent / "model.joblib")

    # Predict and evaluate
    y_pred = model.predict(df[['sentence1', 'sentence2']])
    accuracy = accuracy_score(df['label'], y_pred)
    mcc = matthews_corrcoef(df['label'], y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"MCC: {mcc}")
