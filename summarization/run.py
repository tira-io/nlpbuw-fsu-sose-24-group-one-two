from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from transformers import BertTokenizer, BertModel
import torch

nltk.download('punkt')
nltk.download('stopwords')

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text into words
    words = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Join filtered words back into text
    processed_text = " ".join(filtered_words)
    
    return processed_text

def get_sentence_embeddings(sentences):
    # Encode sentences
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the embeddings from the last hidden layer
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings

def extractive_summarization(text, num_sentences=2):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Check if the preprocessed text is empty
    if not processed_text:
        return ""
    
    # Tokenize the preprocessed text into sentences
    sentences = sent_tokenize(processed_text)
    
    # Ensure there are enough sentences to summarize
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
    
    # Get embeddings for each sentence
    sentence_embeddings = get_sentence_embeddings(sentences)
    
    # Get embeddings for the entire document
    document_embedding = get_sentence_embeddings([processed_text])
    
    # Calculate cosine similarity between each sentence and the document
    similarities = cosine_similarity(sentence_embeddings, document_embedding).flatten()
    
    # Rank sentences based on their similarity scores
    ranked_sentences = [sentence for sentence, similarity in sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True)]
    
    # Select the top-ranked sentences for the summary
    summary = " ".join(ranked_sentences[:num_sentences])
    
    return summary

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")
    print("df size is : ", df.size)
    
    # Apply extractive summarization with BERT embeddings to the records
    df["summary"] = df["story"].apply(lambda x: extractive_summarization(x, num_sentences=2))
    df = df.drop(columns=["story"]).reset_index()
    print(df)
    
    # Save the summarized predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
