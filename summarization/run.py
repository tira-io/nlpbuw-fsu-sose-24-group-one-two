from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string

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
    
    # Tokenize sentences into words
    words_in_sentences = [word_tokenize(sentence) for sentence in sentences]
    
    # Calculate TF-IDF scores for words
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate word scores
    word_scores = {}
    for col in tfidf_matrix.nonzero()[1]:
        word = feature_names[col]
        score = tfidf_matrix[0, col]
        word_scores[word] = score
    
    # Score sentences based on word scores
    sentence_scores = []
    for sentence in words_in_sentences:
        sentence_score = sum(word_scores.get(word, 0) for word in sentence)
        sentence_scores.append(sentence_score)
    
    # Rank sentences based on their scores
    ranked_sentences = [sentence for sentence, score in sorted(zip(sentences, sentence_scores), key=lambda x: x[1], reverse=True)]
    
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
    
    # Apply extractive summarization with text preprocessing to the records
    df["summary"] = df["story"].apply(lambda x: extractive_summarization(x, num_sentences=2))
    df = df.drop(columns=["story"]).reset_index()
    print(df)
    
    # Save the summarized predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
