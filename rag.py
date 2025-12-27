import argparse
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_knowledge(filepath):
    """Loads the knowledge base from a text file, line by line."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Filter out empty lines
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return lines
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

def find_best_match(query, documents):
    """
    Finds the most similar document to the query using TF-IDF and Cosine Similarity.
    """
    # Create a corpus consisting of the documents and the query
    corpus = documents + [query]

    # Vectorize the corpus
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # The usage of the vectorizer handles the math for us.
    # The last vector is our query, the rest are the documents.
    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]

    # Calculate cosine similarity between query and all documents
    # cosine_similarity returns a matrix, we want the first row (comparisons for our query)
    similarities = cosine_similarity(query_vec, doc_vecs).flatten()

    # Find index of best match
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    return best_idx, best_score

def print_banner():
    print("=" * 50)
    print("   DOCU-MIND: Simple RAG System")
    print("=" * 50)

def main():
    print_banner()
    
    knowledge_file = "knowledge.txt"
    documents = load_knowledge(knowledge_file)
    print(f"[INFO] Loaded {len(documents)} facts from {knowledge_file}.\n")

    parser = argparse.ArgumentParser(description="Ask a question to the knowledge base.")
    parser.add_argument("query", nargs="?", help="The question/query to search for.")
    args = parser.parse_args()

    query = args.query
    if not query:
        try:
            query = input("Enter your query: ")
        except KeyboardInterrupt:
            sys.exit(0)

    if not query.strip():
        print("Error: Empty query.")
        return

    print(f"\nSearching for: \"{query}\"...")
    best_idx, score = find_best_match(query, documents)

    print("\n--- Result ---")
    if score > 0.0:
        print(f"Match Score: {score:.4f}")
        print(f"Answer: {documents[best_idx]}")
    else:
        print("No relevant information found in the knowledge base.")
    print("=" * 50)

if __name__ == "__main__":
    main()
