from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pandas as pd
import torch

def load_data(text_path, label_path):
    text_data = pd.read_json(text_path, lines=True)
    label_data = pd.read_json(label_path, lines=True)
    return text_data, label_data

def prepare_predictions(sentences, model, tokenizer, batch_size=32):
    ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
    predictions = []

    # Process in batches
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        ner_results = ner_pipeline(batch)
        
        for j, sentence in enumerate(batch):
            tokens = tokenizer.tokenize(sentence)
            tags = ['O'] * len(tokens)
            for entity in ner_results[j]:
                start, end = entity['start'], entity['end']
                entity_tokens = tokenizer.tokenize(sentence[start:end])
                if entity_tokens:
                    try:
                        start_token_index = tokens.index(entity_tokens[0])
                        tags[start_token_index] = f"B-{entity['entity_group']}"
                        for k in range(1, len(entity_tokens)):
                            tags[start_token_index + k] = f"I-{entity['entity_group']}"
                    except ValueError:
                        pass
            predictions.append(tags)
    return predictions

if __name__ == "__main__":
    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
    targets_validation = tira.pd.truths("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")

    # Limit to first 100 records
    text_validation = text_validation
    targets_validation = targets_validation

    # Construct the local path to the model directory
    model_dir = str(Path(__file__).parent / "local_model")
    
    # Load pre-trained model and tokenizer from local directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    # Prepare predictions
    sentences = text_validation['sentence'].tolist()
    predicted_tags = prepare_predictions(sentences, model, tokenizer)

    # Create the predictions DataFrame
    predictions = pd.DataFrame({'id': text_validation['id'], 'tags': predicted_tags})

    # Save predictions to file
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
