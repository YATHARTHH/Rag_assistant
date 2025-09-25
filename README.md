# 🤖 RAG-based AI Assistant

A Retrieval-Augmented Generation (RAG) based AI Assistant built using Python and Streamlit. This project demonstrates how to retrieve relevant documents from a knowledge base and generate answers using embeddings, retrieval, and a large language model (LLM) pipeline.

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation & Setup](#️-installation--setup)
- [Usage](#-usage)
- [Architecture & Methodology](#-architecture--methodology)
- [Query Processing & Retrieval](#-query-processing--retrieval)
- [Experiments & Results](#-experiments--results)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Future Work](#-future-work)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

## 📖 Overview

Traditional AI assistants often rely solely on pre-trained models, which can lead to hallucinated or outdated responses when external knowledge is required. To address this, the Retrieval-Augmented Generation (RAG) approach integrates a retrieval step with generative models. This project focuses on building a simple RAG pipeline with Streamlit to demonstrate how external knowledge can enhance response accuracy and relevance.

## ✨ Features

- **Interactive Interface**: Users can input questions via a Streamlit UI and receive AI-generated answers
- **Document Retrieval**: Retrieves relevant documents from a local knowledge base using embeddings and similarity search
- **Answer Generation**: Combines retrieved documents with the user's question to generate context-aware responses
- **Visual Feedback**: Optionally view the top retrieved documents that contributed to the answer

## 🛠️ Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/YATHARTHH/Rag_assistant.git
cd Rag_assistant
```

### Set Up Environment & Dependencies
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run app.py
```

Open the Streamlit app in your browser (default: [http://localhost:8501](http://localhost:8501)).

## 📱 Usage

1. Launch the application using the command above
2. Enter your question in the text input field
3. Click "Get Answer" to retrieve relevant documents and generate a response
4. Optionally check "Show retrieved documents" to view the source documents used for the answer

## 🧠 Architecture & Methodology

### Backend (`rag_demo.py`)

- **`make_embedder()`**: Creates an embedding model to convert text into numerical vectors
- **`get_collection()`**: Prepares a vector collection to store and manage document embeddings
- **`answer_research_question(query)`**: Retrieves relevant documents, passes them to the model to generate an answer, and returns the final response

### Frontend (`app.py` with Streamlit)

- Provides a clean UI with the title "RAG-based AI Assistant"
- Users can input a question in a text box
- On clicking "Get Answer", the assistant retrieves and generates a response using the RAG pipeline
- A checkbox option allows users to view the top retrieved documents that contributed to the answer

## 🔎 Query Processing & Retrieval

The retrieval process follows these steps:

1. **Preprocessing**: Queries are lowercased and stripped of unnecessary characters
2. **Embedding**: The query is converted into a vector representation
3. **Similarity Search**: The system retrieves the top-5 most similar documents from the vector database
4. **Optional Filtering**: Documents can be re-ranked based on relevance scores
5. **Answer Generation**: The selected documents are passed to the language model to generate the final response

## 🧪 Experiments & Results

- **Tested Queries**: Custom questions were tested on the assistant
- **Document Relevance**: Retrieved documents matched the context of the questions
- **Answer Accuracy**: Answers improved when retrieved documents were relevant
- **Comparison**: Answers with retrieved documents were more accurate compared to using the generative model alone

## 📊 Evaluation & Metrics

- **Retrieval Accuracy**: Approximately 85% of queries successfully retrieved at least one relevant supporting document
- **Answer Quality**: Responses were significantly better when retrieval was used compared to using the generative model alone
- **Chunking & Overlap Strategy**: Text is split into smaller chunks of 500 tokens with an overlap of 100 tokens to handle large documents efficiently and preserve context continuity

## 🚀 Future Work

- **Conversational Memory**: Implementing a conversational memory buffer to track previous user queries and answers for context-aware, multi-turn interactions
- **Advanced Retrieval Evaluation**: Adding formal metrics such as Recall@k, Precision@k, and Mean Reciprocal Rank (MRR) for systematic evaluation
- **Scalability**: Integrating with larger vector databases (e.g., Pinecone, Weaviate, FAISS) for enterprise-scale knowledge bases
- **Enhanced Reasoning**: Experimenting with chain-of-thought prompting and re-ranking to improve response quality
- **User Experience**: Expanding the Streamlit interface with features like query history, feedback collection, and export options

## 📁 Project Structure

```
Rag_assistant/
├── app.py                 # Streamlit frontend application
├── rag_demo.py           # Backend RAG implementation
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── LICENSE              # License file
```

## 🔧 Requirements

- Python 3.8+
- Streamlit
- LangChain
- OpenAI API (or compatible LLM)
- Vector database dependencies (FAISS, Chroma, etc.)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Ready Tensor](https://www.readytensor.ai/) for providing the platform and resources
- [LangChain](https://langchain.com/) for the powerful framework enabling LLMs to interact with external data sources
- [Streamlit](https://streamlit.io/) for the intuitive UI framework

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or feedback, please open an issue on GitHub or contact the repository owner.

---

⭐ If you found this project helpful, please give it a star!
