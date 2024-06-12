from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import networkx as nx
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize text into words and remove stop words and punctuation
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words and word not in string.punctuation]
    
    # Join filtered words back into text
    processed_text = " ".join(filtered_words)
    
    return processed_text

def extractive_summarization(text, num_sentences=2):
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
    
    # Calculate TF-IDF scores for words
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)
    
    # Debug: Print TF-IDF matrix shape and feature names
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print(f"Feature Names: {vectorizer.get_feature_names_out()}")
    
    # Calculate cosine similarity matrix for sentences
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    # Debug: Print cosine similarity matrix
    print(f"Cosine Similarity Matrix: {sim_matrix}")
    
    # Rank sentences using PageRank algorithm
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    
    # Debug: Print PageRank scores
    print(f"PageRank Scores: {scores}")
    
    # Rank sentences based on their PageRank scores
    ranked_sentences_with_indices = sorted(
        ((scores[i], i, sentence) for i, sentence in enumerate(sentences) if i in scores), 
        key=lambda x: x[0], reverse=True)
    
    # Debug: Print ranked sentences and their indices
    print(f"Ranked Sentences with Indices: {ranked_sentences_with_indices}")
    
    # Select the top-ranked sentences for the summary
    top_sentence_indices = [i for score, i, sentence in ranked_sentences_with_indices[:num_sentences]]
    top_sentences = [sentence for score, i, sentence in ranked_sentences_with_indices[:num_sentences]]
    summary = " ".join(top_sentences)
    
    return summary, top_sentence_indices

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")
    print("df size is : ", df.size)
    
    # Apply extractive summarization with text preprocessing to the records
    df["summary"], df["sentence_indices"] = zip(*df["story"].apply(lambda x: extractive_summarization(x, num_sentences=2)))
    df = df.drop(columns=["story"]).reset_index()
    print(df)
    
    # Save the summarized predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)

    # Print the sentence indices considered for the summary
    for i, row in df.iterrows():
        print(f"ID: {row['id']}, Summary: {row['summary']}, Sentence Indices: {row['sentence_indices']}")
