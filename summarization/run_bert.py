from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import string

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize text into words
    # words = word_tokenize(text)
    
    # # Remove stop words
    # stop_words = set(stopwords.words('english'))
    # filtered_words = [word for word in words if word not in stop_words and word not in string.punctuation]
    
    # # Join filtered words back into text
    # processed_text = " ".join(filtered_words)
    processed_text=text
    
    return processed_text

def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def textrank_sentences_bert(sentences, num_sentences=2):
    if not sentences:
        return "No content available."
    
    # Get BERT embeddings for each sentence
    sentence_embeddings = np.array([get_sentence_embedding(sentence) for sentence in sentences])
    sim_matrix = cosine_similarity(sentence_embeddings)
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)

    # Rank sentences based on PageRank scores
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_sentences = [sentence for score, sentence in ranked_sentences[:num_sentences]]
    summary = ' '.join(top_sentences)
    return summary

def extractive_summarization(text, num_sentences=2):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Tokenize the preprocessed text into sentences
    sentences = sent_tokenize(processed_text)
    
    # Ensure there are enough sentences to summarize
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
    
    return textrank_sentences_bert(sentences, num_sentences)

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")
    print("df size is : ", df.size)
    
    # Apply extractive summarization with text preprocessing to the records
    df["summary"] = df["story"].apply(lambda x: extractive_summarization(x, num_sentences=2))
    df = df.drop(columns=["story"]).reset_index()
    print(df)
    
    # Save the summarized predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
