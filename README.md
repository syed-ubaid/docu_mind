# DocuMind 🧠

> A professional Retrieval-Augmented Generation (RAG) system for extracting insights from your knowledge base.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

DocuMind is a lightweight yet powerful tool that uses TF-IDF and Cosine Similarity to find the most relevant information from a text-based knowledge base. It's designed to be simple to use while providing robust and accurate results.

## 🚀 Features

*   **Efficient Retrieval**: Uses Scikit-Learn's TF-IDF vectorization for fast text matching.
*   **CLI Interface**: Clean and colorful command-line interface.
*   **Extensible**: Easy to add more documents to `knowledge.txt`.
*   **Type Safe**: Built with modern Python type hints.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/docu_mind.git
    cd docu_mind
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📖 Usage

### Interactive Mode
Run the script without arguments to enter the interactive query mode:
```bash
python rag.py
```

### Direct Query
Pass your question directly as an argument:
```bash
python rag.py "What is the capital of France?"
```

## 📂 Configuration

Edit the `knowledge.txt` file to populate your knowledge base. Each line should contain a distinct fact or piece of information.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
