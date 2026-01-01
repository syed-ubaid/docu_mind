import argparse
import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

@dataclass
class SearchResult:
    index: int
    score: float
    content: str

class RAGEngine:
    """
    A simple Retrieval-Augmented Generation (RAG) engine using TF-IDF and Cosine Similarity.
    """

    def __init__(self, knowledge_path: str):
        self.knowledge_path = knowledge_path
        self.documents: List[str] = []
        self._load_knowledge()

    def _load_knowledge(self) -> None:
        """Loads the knowledge base from a text file, line by line."""
        try:
            with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                # Filter out empty lines and strip whitespace
                self.documents = [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            print(f"{Fore.RED}Error: File '{self.knowledge_path}' not found.{Style.RESET_ALL}")
            sys.exit(1)
        except Exception as e:
            print(f"{Fore.RED}Error loading knowledge base: {e}{Style.RESET_ALL}")
            sys.exit(1)

    def search(self, query: str) -> Optional[SearchResult]:
        """
        Finds the most similar document to the query.

        Args:
            query: The search query.

        Returns:
            SearchResult object containing index, score, and content, or None if no docs.
        """
        if not self.documents:
            return None

        # Create a corpus consisting of the documents and the query
        corpus = self.documents + [query]

        # Vectorize the corpus
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            # Handle case where vocabulary is empty
            return None

        # The last vector is our query, the rest are the documents.
        query_vec = tfidf_matrix[-1]
        doc_vecs = tfidf_matrix[:-1]

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, doc_vecs).flatten()

        # Find best match
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        return SearchResult(index=best_idx, score=best_score, content=self.documents[best_idx])

def print_banner():
    print(f"{Fore.CYAN}{'=' * 60}")
    print(f"   DOCU-MIND: Professional RAG System")
    print(f"{'=' * 60}{Style.RESET_ALL}")

def main():
    print_banner()
    
    knowledge_file = "knowledge.txt"
    engine = RAGEngine(knowledge_file)
    print(f"{Fore.GREEN}[INFO] Loaded {len(engine.documents)} facts from {knowledge_file}.{Style.RESET_ALL}\n")

    parser = argparse.ArgumentParser(description="Ask a question to the knowledge base.")
    parser.add_argument("query", nargs="?", help="The question/query to search for.")
    args = parser.parse_args()

    query = args.query
    if not query:
        try:
            query = input(f"{Fore.YELLOW}Enter your query: {Style.RESET_ALL}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)

    if not query.strip():
        print(f"{Fore.RED}Error: Empty query.{Style.RESET_ALL}")
        return

    print(f"\nSearching for: \"{Style.BRIGHT}{query}{Style.RESET_ALL}\"...")
    result = engine.search(query)

    print(f"\n{Fore.CYAN}--- Result ---{Style.RESET_ALL}")
    if result and result.score > 0.1: # Added a small threshold
        print(f"Match Score: {Fore.YELLOW}{result.score:.4f}{Style.RESET_ALL}")
        print(f"Answer: {Style.BRIGHT}{result.content}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}No relevant information found in the knowledge base.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
