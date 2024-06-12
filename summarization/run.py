from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text into words
    words = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Join words back into a single string
    processed_text = ' '.join(words)
    
    return processed_text

def extractive_summarization(text, num_sentences=3):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Check if the preprocessed text is empty
    if not processed_text:
        return "", []
    
    # Tokenize the preprocessed text into sentences
    sentences = sent_tokenize(text)  # Use original text to preserve sentences
    processed_sentences = sent_tokenize(processed_text)
    
    # Ensure there are enough sentences to summarize
    if len(sentences) <= num_sentences:
        return " ".join(sentences), list(range(len(sentences)))
    
    # Calculate TF-IDF scores for words with n-grams (1,2)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)
    
    # Calculate cosine similarity matrix for sentences
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    # Rank sentences using PageRank algorithm
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    
    # Rank sentences based on their PageRank scores
    ranked_sentences_with_indices = sorted(
        ((scores[i], i, sentence) for i, sentence in enumerate(sentences) if i in scores), 
        key=lambda x: x[0], reverse=True)
    
    # Select the top-ranked sentences for the summary
    top_sentence_indices = sorted([i for score, i, sentence in ranked_sentences_with_indices[:num_sentences]])
    top_sentences = [sentences[i] for i in top_sentence_indices]
    summary = " ".join(top_sentences)
    
    return summary, top_sentence_indices

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")
    print("df size is : ", df.size)
    
    # Apply extractive summarization with text preprocessing to the records
    df["summary"], df["sentence_indices"] = zip(*df["story"].apply(lambda x: extractive_summarization(x, num_sentences=3)))
    df = df.drop(columns=["story"]).reset_index()
    
    # Save the summarized predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
