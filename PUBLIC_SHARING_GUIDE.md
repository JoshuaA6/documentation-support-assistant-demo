# Public Sharing Guide

This project is easiest to share safely by keeping two versions:

- A private demo repo with the live app, scraped docs, and vector database
- A public portfolio repo with only your code, setup docs, and placeholder data instructions

## Immediate Recommendation

If your current GitHub repo is public, make it private before sharing it further. This repo currently contains or has contained:

- scraped help-center article JSON files under `scraped_docs/`
- a derived vector database under `vector_db/`
- a targeted scraper script for the Klaviyo help center

Even if you stop tracking those files later, they may still exist in prior commits in the repo history.

## Safe Public Repo Contents

Keep these:

- Streamlit UI code
- RAG pipeline code
- embedding pipeline structure
- dependency files
- `.env.example`
- architecture notes
- screenshots you created
- your own README and portfolio write-up

Remove these:

- `scraped_docs/`
- `vector_db/`
- `docs_index.json`
- any article text copied from Klaviyo
- any generated files derived from copied Klaviyo content
- secrets, keys, `.env`, and Streamlit secrets files

Be cautious about publishing:

- `scrape_klaviyo_help_center.py`

If you keep scraper code public at all, make it generic rather than framed as a scraper specifically for Klaviyo.

## Recommended Public Disclaimer

Use wording like:

> Unofficial demo project. Not affiliated with, endorsed by, or sponsored by Klaviyo.

## Recommended Naming

Safer names:

- `Documentation Support Assistant Demo`
- `RAG Help Center Assistant`
- `Support Knowledge Base Chatbot`

Riskier names:

- names that imply the app is an official Klaviyo product

## Best Portfolio Setup

1. Keep this repo private for demos.
2. Generate a sanitized portfolio copy with `python3 prepare_public_portfolio.py`.
3. Create a separate public GitHub repo from `public_portfolio_bundle/`.
4. Use that sanitized repo in LinkedIn, resume, and GitHub links.
5. Demo the private deployment when someone wants to see the real working version.

## Demo Access Without Public Risk

You can still demo the project easily by:

- screen sharing the private Streamlit app
- sharing a limited-access private deployment link
- recording a short walkthrough video

That gives you a public-safe portfolio artifact and a private live demo at the same time.
