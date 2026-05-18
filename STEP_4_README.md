# Step 4: Build the RAG Pipeline (Retrieval-Augmented Generation)

## 🎯 What is RAG?

**RAG** = **R**etrieval + **A**ugmentation + **G**eneration

It's the magic that makes your AI Support Agent work:

```
User Question
    ↓
Retrieve relevant docs from ChromaDB (Step 3)
    ↓
Augment LLM prompt with retrieved docs
    ↓
Generate answer using GPT-5.4-mini
    ↓
Extract citations + follow-up questions
    ↓
Display to user
```

---

## 📁 Files Created

### 1. **step4_rag_pipeline.py** (Core RAG)
The brain of your support agent.

**Key class: `RAGPipeline`**

Methods:
- `retrieve_context(query)` - Find relevant docs from ChromaDB
- `generate_answer(query, context)` - LLM generates answer
- `generate_followups(query, answer)` - Suggest related questions
- `answer_question(query)` - Full pipeline (retrieve → augment → generate)

**How it works:**
```python
from step4_rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline(top_k=3)  # Retrieve top 3 chunks

# Ask a question
result = rag.answer_question("How do I create a flow?")

# Result contains:
# - answer: LLM-generated response
# - sources: List of citations
# - followup_questions: Suggested next questions
```

### 2. **step4b_streamlit_ui.py** (Interactive Chatbot)
Beautiful web interface for your chatbot.

**Features:**
- ✨ Clean, modern chat interface
- 🎯 Beginner mode (step-by-step explanations)
- 📚 Click to view sources
- 💡 One-click follow-up questions
- ⚙️ Adjustable settings
- 🗑️ Clear chat history

Run with:
```bash
streamlit run step4b_streamlit_ui.py
```

Opens at: `http://localhost:8501`

### 3. **requirements.txt** (Updated)
Added:
- `streamlit==1.28.1` - Web UI
- `streamlit-chat==0.1.1` - Chat components

---

## 🚀 How to Run

### Option 1: Test RAG Pipeline (Terminal)

```bash
python3 step4_rag_pipeline.py
```

This runs demo queries and shows:
- Retrieved sources
- Generated answers
- Citations
- Follow-up questions

**Example output:**
```
🔍 Query: How do I create a flow?
📚 Retrieving relevant documentation...
   Found 3 relevant sources
🤖 Generating answer with LLM...
✓ Answer generated
💡 Generating follow-up questions...
✓ Follow-ups generated

📋 ANSWER:
To create a flow in Klaviyo, you need to...

📚 SOURCES USED:
   1. 1.How to use Flows AI to build a flow | Klaviyo Help Center.pdf
      → Type out a description of the flow you want to create...
```

### Option 2: Interactive Web UI (Recommended!)

```bash
pip install -r requirements.txt
streamlit run step4b_streamlit_ui.py
```

Then:
1. Open `http://localhost:8501` in your browser
2. Type a question
3. Get an instant answer with citations
4. Click follow-up questions to explore more
5. Toggle "Beginner Mode" for simpler explanations

---

## 🔍 Understanding the Pipeline

### Step 1: Retrieve
```python
query = "How do I create a flow?"
query_embedding = generate_embedding(query)
results = chromadb.query(query_embedding, top_k=3)
# Returns top 3 most similar chunks from docs
```

**Why embedding-based search?**
- Finds *meaning*, not just keywords
- "How do I create a flow?" matches "To create a flow..." even with different words
- Semantic search >> keyword search

### Step 2: Augment
```python
context = """
[Source 1: How to use Flows AI to build a flow]
Type out a description of the flow you want to create...

[Source 2: How to make sure your flow is ready...]
Before you begin, please see our guide on how to get started...
"""

# Build LLM prompt with context
prompt = system_prompt + context + user_query
```

**Why augment?**
- LLM has no knowledge of Klaviyo docs
- Context grounds the LLM in accurate information
- Reduces hallucinations

### Step 3: Generate
```python
response = client.chat.completions.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
)
answer = response.choices[0].message.content
```

**Result:**
- Accurate, cited answers
- Follow-up questions
- Professional tone

---

## 💡 Customization

### Change Top-K Results
In `step4_rag_pipeline.py`:
```python
rag = RAGPipeline(top_k=5)  # Get 5 sources instead of 3
```

More sources = more context but less focused
Fewer sources = focused but might miss info

### Adjust LLM Temperature
In `step4_rag_pipeline.py`, find `generate_answer()`:
```python
response = client.chat.completions.create(
    temperature=0.7,  # Change this
    # 0.0 = deterministic, 1.0 = creative
)
```

### Customize System Prompt
In `RAGPipeline.__init__()`:
```python
self.system_prompt = """Your custom instructions here..."""
```

Add instructions for:
- Tone (professional, friendly, technical)
- Format (bullet points, step-by-step, code samples)
- Behavior (when to admit not knowing something, etc.)

---

## 🧪 Testing

### Test 1: Basic RAG Pipeline
```bash
python3 step4_rag_pipeline.py
```

Check:
- ✓ Answers are based on retrieved docs
- ✓ Citations are accurate
- ✓ Follow-ups are relevant

### Test 2: Streamlit UI
```bash
streamlit run step4b_streamlit_ui.py
```

Test:
- ✓ Chat works
- ✓ Can view sources
- ✓ Follow-up buttons work
- ✓ Beginner mode toggle works

### Test 3: Custom Queries
Try questions like:
- "How do I create a segment-triggered flow?"
- "What's the best way to test my flow?"
- "Can you explain conditional splits?"
- "How do I monitor flow performance?"

---

## 📊 Architecture Overview

```
Your AI Support Agent
├── User Interface
│   └── Streamlit Web UI (step4b_streamlit_ui.py)
│       ↓
├── RAG Pipeline (step4_rag_pipeline.py)
│   ├── Retrieve
│   │   └── ChromaDB (vector database)
│   ├── Augment
│   │   └── Context building
│   └── Generate
│       └── Azure OpenAI (gpt-5.4-mini)
│           ↓
└── Knowledge Base
    └── 112 embedded chunks from 10 PDFs
```

---

## ✨ Features Explained

### 1. **Citations** 🔗
Every answer shows which docs were used
- Builds trust
- Lets users verify answers
- Shows Responsible AI

### 2. **Follow-up Questions** 💡
AI suggests what to ask next
- Improves UX
- Encourages exploration
- Shows understanding of context

### 3. **Beginner Mode** 📖
Simpler language + step-by-step format
- Good for new users
- Professional presentation
- Personalization example

### 4. **Source Filtering** 📚
Easy way to see which docs answered your question
- Transparency
- Trust building
- Documentation exploration

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install -r requirements.txt
```

### "DeploymentNotFound" error
- Check `.env` file has correct credentials
- Make sure `text-embedding-3-small` deployment exists in Azure

### Answers are not from docs
- Check that Step 3 (embeddings pipeline) completed successfully
- Verify ChromaDB folder and files exist
- Try increasing `top_k` to get more context

### Streamlit won't start
```bash
streamlit run step4b_streamlit_ui.py --logger.level=debug
```

---

## 🎯 Success Checklist

- [ ] `requirements.txt` has streamlit dependencies
- [ ] `step4_rag_pipeline.py` created successfully
- [ ] `step4b_streamlit_ui.py` created successfully
- [ ] Run `python3 step4_rag_pipeline.py` with demo queries
- [ ] Run `streamlit run step4b_streamlit_ui.py` successfully
- [ ] Ask a question and get a relevant answer
- [ ] Click a follow-up question and get a new answer
- [ ] View sources for an answer
- [ ] Toggle beginner mode

---

## 🚀 What's Next?

Once Step 4 is working, you can:

1. **Deploy to production** - Host Streamlit app on cloud
2. **Add more docs** - Scale to full Klaviyo documentation
3. **Add user feedback** - Track which answers help/don't help
4. **A/B test prompts** - Test different system prompts
5. **Monitor usage** - Analytics on what users ask
6. **Fine-tune LLM** - Custom model based on Klaviyo domain
7. **Add multi-language** - Support for different languages
8. **Integrate with Slack/Teams** - Embed bot in chat platforms

---

## 📚 Key Concepts

**Embedding**: Vector representation of text (high-dimensional)
**Vector Database**: Stores embeddings for fast similarity search
**RAG**: Retrieval-Augmented Generation (retrieve docs → augment prompt → generate answer)
**Citations**: References to source documents
**LLM**: Large Language Model (gpt-5.4-mini)
**Context Window**: Maximum tokens the LLM can read at once

---

## 📖 Full Example

```python
from step4_rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline(top_k=3)

# Ask questions
queries = [
    "How do I create a flow?",
    "What are conditional splits?",
    "How do I test a flow before sending?"
]

for query in queries:
    result = rag.answer_question(query, verbose=True)
    
    print(f"\nQ: {query}")
    print(f"A: {result['answer']}")
    print(f"\nSources:")
    for source in result['sources']:
        print(f"  - {source['source']}")
    print(f"\nFollow-ups:")
    for followup in result['followup_questions']:
        print(f"  - {followup}")
```

---

## 🎉 Congratulations!

You've built a production-ready AI Support Agent:

✅ **Step 1**: Understand the concept
✅ **Step 2**: Collect & prepare data
✅ **Step 3**: Extract, embed, store in vector DB
✅ **Step 4**: Build RAG pipeline with UI

You now have:
- Semantic search over documentation
- LLM-powered answers with citations
- Interactive web interface
- Follow-up suggestions
- Beginner-friendly explanations

This is the exact pattern used by:
- ChatGPT Retrieval
- GitHub Copilot Chat
- Microsoft Copilot
- All modern AI support tools

**Ready to ship! 🚀**
