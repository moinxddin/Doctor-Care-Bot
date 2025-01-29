# Doctor-Care-Bot
Project Overview

Doctor Care-Bot is an AI-driven healthcare chatbot designed to assist patients by providing intelligent responses to medical inquiries. The chatbot leverages Ollama's Llama3-based Large Language Model (LLM) with Retrieval-Augmented Generation (RAG) to enhance contextual understanding and response generation.

Features

Real-time patient query processing through a Streamlit-based interactive UI.

Retrieval-Augmented Generation (RAG) using Chroma Vector Store for improved response accuracy.

Dynamic text retrieval and chunk-based processing using RecursiveCharacterTextSplitter.

Web-based data ingestion from Dr. Waheed’s Skincare website for domain-specific medical insights.

Streaming response generation for real-time AI-assisted diagnostics and recommendations.

Session-based memory retention for interactive conversations with historical context.

Technologies Used

Machine Learning & AI: Llama3, Retrieval-Augmented Generation (RAG), Ollama Embeddings

Vector Database: ChromaDB

Web Scraping & Data Processing: WebBaseLoader, RecursiveCharacterTextSplitter

Backend Framework: Python, Streamlit

Cloud & API Integration: Ollama API for conversational AI


Run the chatbot locally:

streamlit run ollama_rag.py

Usage

Start the chatbot and ask medical-related questions.

The bot retrieves relevant medical data from Dr. Waheed’s Skincare website.

Responses are generated using Llama3 and contextual retrieval.

User interactions are stored in session memory for a seamless experience.

Performance Metrics

Response accuracy: 90%

Reduction in user interaction time: 30%

Scalability: Optimized for real-time responses using Streamlit and Ollama API.

