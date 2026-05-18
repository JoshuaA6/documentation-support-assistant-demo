# Step 3: Extract Docs → Embed → Store in Vector DB

## 📁 Files Created

### 1. **step3_embeddings_pipeline.py** (Main Script)
This is your core Step 3 implementation. It does everything:

```
PDFs in pdfs/ 
  ↓
Extract Text (PyPDF2)
  ↓
Chunk Text (500-1000 chars with overlap)
  ↓
Generate Embeddings (Azure OpenAI)
  ↓
Store in ChromaDB (Local Vector DB)
  ↓
docs_index.json (metadata index)
```

**What it does:**
- Reads all PDFs from `pdfs/` folder
- Extracts text from each page
- Intelligently chunks text (keeps sentences together)
- Generates embeddings using `text-embedding-3-small`
- Stores embeddings + metadata in ChromaDB
- Creates `docs_index.json` with references

### 2. **step3b_test_retrieval.py** (Testing Script)
Tests that everything works. Run this after the main pipeline.

Shows you:
- How to query your vector DB
- How to retrieve similar docs (this is RAG!)
- Example output with confidence scores

### 3. **.env** (Your Credentials)
Stores your Azure OpenAI configuration:
- Endpoint
- API Key
- Deployment name

⚠️ **Keep this file safe!** Don't commit to GitHub.

### 4. **requirements.txt** (Dependencies)
Install with: `pip install -r requirements.txt`

---

## 🚀 How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the embedding pipeline
```bash
python step3_embeddings_pipeline.py
```

This will:
- Extract text from all 10 PDFs
- Create ~100-200 chunks
- Generate embeddings for each
- Store everything in ChromaDB (in `vector_db/` folder)
- Create `docs_index.json`

**Expected output:**
```
Found 10 PDF files
--------------------------------------------------

Processing: 1.How to use Flows AI to build a flow | Klaviyo Help Center.pdf
  ✓ Extracted 12500 characters
  ✓ Created 15 chunks
  ✓ Stored all chunks in ChromaDB
...

==================================================
✅ Pipeline Complete!
   • Total chunks processed: 156
   • ChromaDB stored at: vector_db
   • Index saved to: docs_index.json
==================================================
```

### Step 3: Test your retrieval system
```bash
python step3b_test_retrieval.py
```

This will:
- Query your vector DB with sample questions
- Show you the top 3 most relevant docs
- Display confidence scores

**Expected output:**
```
📝 Query: 'How do I create a flow?'
------

1. From: 1.How to use Flows AI to build a flow | Klaviyo Help Center.pdf
   Preview: To create a flow in Klaviyo, you need to...
   Confidence: 0.1234

2. From: 4. How to create a segment- or list-triggered flow...
   Preview: A segment-triggered flow allows you to...
   Confidence: 0.2456
...
```

---

## 📊 What Gets Created

After running the pipeline:

```
your_project/
├── vector_db/           ← ChromaDB database (local storage)
│   ├── chroma.sqlite3
│   └── ...
├── docs_index.json      ← Metadata index (human-readable)
└── [other files]
```

---

## 🔍 Understanding Each Component

### Text Extraction (PyPDF2)
Reads PDF files and extracts all text. Simple but works great.

### Chunking Strategy
- **Goal**: Break docs into bite-sized pieces
- **Size**: 800 characters per chunk (with 200 char overlap)
- **Method**: Split by sentences first (keeps context intact)
- **Benefit**: Better search results + faster embeddings

### Embeddings (Azure OpenAI)
- **Model**: `text-embedding-3-small` (fast, efficient)
- **Output**: 1536-dimensional vector per chunk
- **Cost**: Very cheap (~$0.02 per 1M tokens)
- **What it means**: Each chunk is a point in vector space

### ChromaDB (Vector Database)
- **Type**: Local, open-source vector database
- **Storage**: Persistent (survives restarts)
- **Search**: Uses cosine similarity
- **Cost**: Free
- **Benefit**: Perfect for prototypes & testing

### Metadata Tracking
Each chunk stores:
- `source`: PDF name (for citations)
- `pdf_name`: Full filename
- `chunk_index`: Position in doc
- `total_chunks_in_doc`: Total chunks from this doc

---

## 🔧 Customization

### Change chunk size:
In `step3_embeddings_pipeline.py`, find:
```python
chunks = chunk_text(text, chunk_size=800, overlap=200)
```

Adjust `chunk_size` (bigger = fewer chunks, broader context)

### Change number of results:
In `step3b_test_retrieval.py`:
```python
results = retrieve_similar_docs(query, top_k=3)  # Change 3 to 5, 10, etc.
```

### Use different embedding model:
In `.env`, change:
```
AZURE_OPENAI_DEPLOYMENT_NAME=text-embedding-ada-002
```

---

## ✅ What's Next (Step 4: Build RAG)

Once Step 3 is complete, you have:
- ✅ Embedded chunks in ChromaDB
- ✅ Ability to search & retrieve
- ✅ Metadata tracking

Next, Step 4 will:
1. Take user queries
2. Retrieve top-k similar chunks (from this step!)
3. Feed them to GPT-5.4-mini
4. Generate citations
5. Suggest follow-ups

**You already have the retrieval pipeline ready! 🎉**

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'PyPDF2'"
```bash
pip install -r requirements.txt
```

### "Error generating embedding"
Check:
- Is `.env` file present and filled?
- Is your API key correct?
- Is your deployment name `text-embedding-3-small`?

### "No PDF files found in pdfs"
Make sure your PDFs are in the `pdfs/` folder (they should be!)

### "ChromaDB persists but queries return nothing"
Delete `vector_db/` folder and re-run the pipeline.

---

## 📚 Key Concepts

**RAG (Retrieval-Augmented Generation):**
- **Retrieval**: Find relevant docs (what you just built!)
- **Augmented**: Add them to the LLM prompt
- **Generation**: LLM answers using those docs

**Vector Embeddings:**
- Convert text → numbers
- Similar text → similar numbers
- Allows semantic search (meaning-based, not keyword-based)

**ChromaDB:**
- Stores vectors + metadata
- Fast similarity search
- No server needed

---

## 🎯 Success Checklist

- [ ] `requirements.txt` installed
- [ ] `.env` file has your credentials
- [ ] `pdfs/` folder has 10 Klaviyo docs
- [ ] Run `python step3_embeddings_pipeline.py` successfully
- [ ] `vector_db/` folder created
- [ ] `docs_index.json` created with ~150+ entries
- [ ] Run `python step3b_test_retrieval.py` and see results
- [ ] Test queries return relevant docs

Once all ✅, you're ready for Step 4!
