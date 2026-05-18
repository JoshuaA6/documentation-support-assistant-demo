# Documentation Support Assistant Demo

Unofficial demo project. Not affiliated with, endorsed by, or sponsored by Klaviyo.

## Overview

This repository contains a portfolio-safe version of a documentation support assistant built with Streamlit, Azure OpenAI, and vector search.

The project simulates a modern help-center experience: a user asks a product question, the app retrieves relevant documentation chunks, and the assistant responds with grounded answers, citations, and suggested follow-up prompts.

## Why I Built It

I wanted to demonstrate practical retrieval-augmented generation work beyond a basic chatbot UI. The goal was to build something closer to a real support workflow, including:

- source-aware answers
- follow-up question generation
- multilingual chat support
- retrieval quality handling
- end-to-end debugging and deployment work

## Core Capabilities

- Streamlit chat interface for a lightweight support experience
- retrieval-augmented generation over a documentation-style knowledge base
- source citations and related follow-up prompts
- Azure OpenAI-based answer generation and embeddings
- local vector search with ChromaDB
- modular ingestion, retrieval, and UI layers

## Architecture

At a high level, the system works like this:

1. Documentation content is prepared and chunked for retrieval.
2. Chunks are embedded and stored in a local vector database.
3. A user submits a question through the Streamlit UI.
4. The app retrieves the most relevant chunks.
5. The LLM generates a grounded answer using the retrieved context.
6. The UI renders the response, citations, and suggested follow-up questions.

## Repository Contents

This public repository includes:

- Streamlit UI code
- RAG pipeline code
- embeddings/indexing pipeline structure
- setup notes and environment-variable template

This public repository intentionally excludes:

- third-party scraped documentation
- derived vector databases
- copied article text
- local secrets and deployment credentials

To run the full pipeline end-to-end, provide your own supported document corpus and generate your own embeddings index.

## Tech Stack

- Python
- Streamlit
- Azure OpenAI
- ChromaDB

## Local Setup

```bash
pip install -r requirements.txt
cp .env.example .env
streamlit run main.py
```

Note: this portfolio-safe version does not include the original documentation corpus or vector index, so you will need to supply your own data source before the chatbot can answer questions end-to-end.

## What This Project Demonstrates

- applied RAG system design
- LLM application development with citations and retrieval grounding
- support-oriented UX decisions for chat workflows
- debugging across dependencies, data pipelines, and deployment entrypoints

## Project Notes

This repo is intentionally structured as a clean portfolio artifact. It focuses on the architecture, implementation approach, and product thinking behind the assistant rather than redistributing third-party documentation content.
