# Documentation Support Assistant Demo

Unofficial demo project. Not affiliated with, endorsed by, or sponsored by Klaviyo.

## Overview

This repository contains a portfolio-safe version of a documentation support assistant built with Streamlit, Azure OpenAI, and a local vector-search workflow.

The project demonstrates:

- retrieval-augmented generation
- citation-aware answers
- follow-up question suggestions
- multilingual chat UX
- practical deployment and debugging work

## What's Included

- Streamlit UI code
- RAG pipeline code
- environment-variable template
- setup and architecture notes

## What's Not Included

This public version intentionally excludes any third-party scraped documentation, derived vector databases, or private deployment secrets.

To run the full pipeline, supply your own documentation corpus and build your own embeddings index.

## Example Tech Stack

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

You will need to provide your own supported document corpus and index before the app can answer questions end-to-end.
