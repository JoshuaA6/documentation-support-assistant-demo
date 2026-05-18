"""
Step 4B: Streamlit UI for AI Support Agent Chatbot

Interactive web interface for the RAG pipeline.
Run with: streamlit run step4b_streamlit_ui.py
"""

import json
import html
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from step4_rag_pipeline import RAGPipeline
import re

# Page config
st.set_page_config(
    page_title="Klaviyo AI Support Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    :root {
        --page-bg: #0e1117;
        --text-primary: #111827;
        --text-secondary: #f8fafc;
        --user-bg: #dbeafe;
        --user-border: #2563eb;
        --assistant-bg: #111111;
        --assistant-border: #22c55e;
    }
    .main {
        padding: 2rem;
    }
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .language-picker-wrap {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 0.5rem;
    }
    .assistant-meta {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        margin-bottom: 0.65rem;
        padding: 0.3rem 0.6rem;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.14);
        color: #d1fae5 !important;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .retrieval-warning {
        margin: 0 0 0.75rem 0;
        padding: 0.75rem 0.9rem;
        border-radius: 0.65rem;
        background: rgba(251, 191, 36, 0.14);
        border: 1px solid rgba(251, 191, 36, 0.4);
        color: #fef3c7 !important;
        font-size: 0.92rem;
    }
    .message-container {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        line-height: 1.6;
        font-size: 1rem;
    }
    .message-container strong,
    .message-container p,
    .message-container li,
    .message-container span,
    .message-container div {
        color: inherit !important;
    }
    .user-message {
        background-color: var(--user-bg);
        border-left: 4px solid var(--user-border);
        color: var(--text-primary) !important;
    }
    .assistant-message {
        background-color: var(--assistant-bg);
        border-left: 4px solid var(--assistant-border);
        color: var(--text-secondary) !important;
    }
    .user-message strong,
    .assistant-message strong {
        color: var(--text-secondary) !important;
    }
    .user-message strong {
        color: var(--text-primary) !important;
    }
    .source-badge {
        display: inline-block;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        font-size: 0.9rem;
        color: var(--text-primary) !important;
    }
    .source-badge a {
        color: #1d4ed8 !important;
        font-weight: 600;
        text-decoration: underline;
    }
    .source-meta {
        display: block;
        margin-top: 0.3rem;
        color: #4b5563 !important;
        font-size: 0.8rem;
    }
    .starter-card {
        padding: 1rem;
        border-radius: 0.9rem;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(16, 185, 129, 0.1));
        border: 1px solid rgba(148, 163, 184, 0.22);
        margin: 0.5rem 0 1rem 0;
    }
    .starter-card h3,
    .starter-card p {
        color: #e5eefc !important;
    }
    .followup-button {
        margin: 0.5rem;
        padding: 0.5rem 1rem;
        background-color: #e0e0e0;
        border: 1px solid #999;
        border-radius: 0.25rem;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

LOG_DIR = Path(tempfile.gettempdir()) / "klaviyo_agent_logs"

LANGUAGE_OPTIONS = {
    "EN": {
        "name": "English",
        "llm_name": "English",
        "header_title": "Klaviyo AI Support Agent",
        "header_subtitle": "Ask questions about Klaviyo and get instant answers with citations across flows, campaigns, billing, integrations, SMS, forms, and more.",
        "settings": "Settings",
        "beginner_mode": "Beginner Mode",
        "beginner_help": "Get step-by-step explanations in simple language",
        "show_sources": "Show Sources",
        "show_sources_help": "Display citation sources below each answer",
        "num_sources": "Number of sources to retrieve",
        "num_sources_help": "More sources = more context but potentially less focused",
        "clear_history": "Clear Chat History",
        "example_questions": "Example Questions",
        "chat_title": "Chat",
        "you": "You",
        "assistant": "Klaviyo Assistant",
        "sources": "Sources",
        "source_label": "Source",
        "followups": "Suggested Follow-ups:",
        "ask_prompt": "Ask a question about Klaviyo...",
        "ask_placeholder": "e.g., How do I create and send an email campaign?",
        "send": "Send",
        "searching": "Searching documentation...",
        "step_prefix": "Step-by-step guide:",
        "footer": "Powered by Azure OpenAI | Built with Streamlit | Data from Klaviyo Documentation",
        "language_label": "Language",
        "starter_title": "Start with one of these prompts",
        "starter_body": "Use a starter prompt or ask your own question. The assistant will answer using Klaviyo documentation and show the sources it relied on.",
        "answer_language_badge": "answer",
        "limited_answer_warning": "This answer may be limited because the closest matching docs were weaker or less diverse than usual.",
    },
    "ES": {
        "name": "Espanol",
        "llm_name": "Spanish",
        "header_title": "Agente de Soporte IA de Klaviyo",
        "header_subtitle": "Haz preguntas sobre Klaviyo y recibe respuestas instantaneas con citas sobre flujos, campanas, facturacion, integraciones, SMS, formularios y mas.",
        "settings": "Configuracion",
        "beginner_mode": "Modo Principiante",
        "beginner_help": "Recibe explicaciones paso a paso en lenguaje sencillo",
        "show_sources": "Mostrar Fuentes",
        "show_sources_help": "Muestra las fuentes citadas debajo de cada respuesta",
        "num_sources": "Numero de fuentes a recuperar",
        "num_sources_help": "Mas fuentes = mas contexto pero potencialmente menos enfoque",
        "clear_history": "Borrar Historial",
        "example_questions": "Preguntas de Ejemplo",
        "chat_title": "Chat",
        "you": "Tu",
        "assistant": "Asistente de Klaviyo",
        "sources": "Fuentes",
        "source_label": "Fuente",
        "followups": "Preguntas sugeridas:",
        "ask_prompt": "Haz una pregunta sobre Klaviyo...",
        "ask_placeholder": "p. ej., Como creo y envio una campana de correo electronico?",
        "send": "Enviar",
        "searching": "Buscando en la documentacion...",
        "step_prefix": "Guia paso a paso:",
        "footer": "Impulsado por Azure OpenAI | Creado con Streamlit | Datos de la documentacion de Klaviyo",
        "language_label": "Idioma",
        "starter_title": "Empieza con una de estas preguntas",
        "starter_body": "Usa una pregunta sugerida o escribe la tuya. El asistente respondera usando la documentacion de Klaviyo y mostrara las fuentes utilizadas.",
        "answer_language_badge": "respuesta",
        "limited_answer_warning": "Esta respuesta puede ser limitada porque los documentos mas cercanos fueron menos relevantes o menos variados de lo habitual.",
    },
    "FR": {
        "name": "Francais",
        "llm_name": "French",
        "header_title": "Agent de Support IA Klaviyo",
        "header_subtitle": "Posez des questions sur Klaviyo et obtenez des reponses instantanees avec citations sur les flows, campagnes, facturation, integrations, SMS, formulaires et plus encore.",
        "settings": "Parametres",
        "beginner_mode": "Mode Debutant",
        "beginner_help": "Obtenez des explications pas a pas dans un langage simple",
        "show_sources": "Afficher les Sources",
        "show_sources_help": "Affiche les sources citees sous chaque reponse",
        "num_sources": "Nombre de sources a recuperer",
        "num_sources_help": "Plus de sources = plus de contexte mais potentiellement moins de precision",
        "clear_history": "Effacer l'Historique",
        "example_questions": "Questions d'Exemple",
        "chat_title": "Chat",
        "you": "Vous",
        "assistant": "Assistant Klaviyo",
        "sources": "Sources",
        "source_label": "Source",
        "followups": "Questions suggerees :",
        "ask_prompt": "Posez une question sur Klaviyo...",
        "ask_placeholder": "ex. : Comment creer et envoyer une campagne e-mail ?",
        "send": "Envoyer",
        "searching": "Recherche dans la documentation...",
        "step_prefix": "Guide pas a pas :",
        "footer": "Propulse par Azure OpenAI | Construit avec Streamlit | Donnees issues de la documentation Klaviyo",
        "language_label": "Langue",
        "starter_title": "Commencez avec l'une de ces questions",
        "starter_body": "Utilisez une suggestion ou posez votre propre question. L'assistant repondra a partir de la documentation Klaviyo et affichera les sources utilisees.",
        "answer_language_badge": "reponse",
        "limited_answer_warning": "Cette reponse peut etre limitee car les documents les plus proches etaient moins pertinents ou moins varies que d'habitude.",
    },
    "DE": {
        "name": "Deutsch",
        "llm_name": "German",
        "header_title": "Klaviyo KI Support Agent",
        "header_subtitle": "Stelle Fragen zu Klaviyo und erhalte sofortige Antworten mit Quellenangaben zu Flows, Kampagnen, Abrechnung, Integrationen, SMS, Formularen und mehr.",
        "settings": "Einstellungen",
        "beginner_mode": "Einsteiger-Modus",
        "beginner_help": "Erhalte Schritt-fur-Schritt-Erklarungen in einfacher Sprache",
        "show_sources": "Quellen Anzeigen",
        "show_sources_help": "Zeigt Quellen unter jeder Antwort an",
        "num_sources": "Anzahl der Quellen",
        "num_sources_help": "Mehr Quellen = mehr Kontext, aber moglicherweise weniger Fokus",
        "clear_history": "Chatverlauf Loschen",
        "example_questions": "Beispielfragen",
        "chat_title": "Chat",
        "you": "Du",
        "assistant": "Klaviyo Assistent",
        "sources": "Quellen",
        "source_label": "Quelle",
        "followups": "Empfohlene Anschlussfragen:",
        "ask_prompt": "Stelle eine Frage zu Klaviyo...",
        "ask_placeholder": "z. B. Wie erstelle und sende ich eine E-Mail-Kampagne?",
        "send": "Senden",
        "searching": "Dokumentation wird durchsucht...",
        "step_prefix": "Schritt-fur-Schritt-Anleitung:",
        "footer": "Bereitgestellt von Azure OpenAI | Erstellt mit Streamlit | Daten aus der Klaviyo-Dokumentation",
        "language_label": "Sprache",
        "starter_title": "Starte mit einer dieser Fragen",
        "starter_body": "Nutze eine vorgeschlagene Frage oder stelle deine eigene. Der Assistent antwortet anhand der Klaviyo-Dokumentation und zeigt die verwendeten Quellen an.",
        "answer_language_badge": "Antwort",
        "limited_answer_warning": "Diese Antwort kann eingeschrankt sein, weil die am besten passenden Dokumente schwacher oder weniger vielfaltig als ublich waren.",
    },
    "PT": {
        "name": "Portugues",
        "llm_name": "Portuguese",
        "header_title": "Agente de Suporte IA da Klaviyo",
        "header_subtitle": "Faca perguntas sobre a Klaviyo e receba respostas instantaneas com citacoes sobre fluxos, campanhas, cobranca, integracoes, SMS, formularios e muito mais.",
        "settings": "Configuracoes",
        "beginner_mode": "Modo Iniciante",
        "beginner_help": "Receba explicacoes passo a passo em linguagem simples",
        "show_sources": "Mostrar Fontes",
        "show_sources_help": "Exibe as fontes citadas abaixo de cada resposta",
        "num_sources": "Numero de fontes para recuperar",
        "num_sources_help": "Mais fontes = mais contexto, mas potencialmente menos foco",
        "clear_history": "Limpar Historico",
        "example_questions": "Perguntas de Exemplo",
        "chat_title": "Chat",
        "you": "Voce",
        "assistant": "Assistente Klaviyo",
        "sources": "Fontes",
        "source_label": "Fonte",
        "followups": "Sugestoes de proximas perguntas:",
        "ask_prompt": "Faca uma pergunta sobre a Klaviyo...",
        "ask_placeholder": "ex.: Como crio e envio uma campanha de email?",
        "send": "Enviar",
        "searching": "Pesquisando na documentacao...",
        "step_prefix": "Guia passo a passo:",
        "footer": "Desenvolvido com Azure OpenAI | Criado com Streamlit | Dados da documentacao da Klaviyo",
        "language_label": "Idioma",
        "starter_title": "Comece com uma destas perguntas",
        "starter_body": "Use uma sugestao ou faca sua propria pergunta. O assistente respondera com base na documentacao da Klaviyo e mostrara as fontes utilizadas.",
        "answer_language_badge": "resposta",
        "limited_answer_warning": "Esta resposta pode ser limitada porque os documentos mais proximos estavam mais fracos ou menos diversos do que o normal.",
    },
}

EXAMPLE_QUESTIONS = {
    "EN": [
        "How do I create and send an email campaign?",
        "How does Klaviyo billing work?",
        "How do I set up SMS in Klaviyo?",
        "How do sign-up forms work?",
        "How do I connect Shopify to Klaviyo?",
        "How do I create an abandoned cart flow?",
        "How do I import subscribers to a list?",
        "How do I manage my account's API keys?",
    ],
    "ES": [
        "Como creo y envio una campana de correo electronico?",
        "Como funciona la facturacion de Klaviyo?",
        "Como configuro SMS en Klaviyo?",
        "Como funcionan los formularios de registro?",
        "Como conecto Shopify con Klaviyo?",
        "Como creo un flujo de carrito abandonado?",
        "Como importo suscriptores a una lista?",
        "Como administro las claves API de mi cuenta?",
    ],
    "FR": [
        "Comment creer et envoyer une campagne e-mail ?",
        "Comment fonctionne la facturation Klaviyo ?",
        "Comment configurer le SMS dans Klaviyo ?",
        "Comment fonctionnent les formulaires d'inscription ?",
        "Comment connecter Shopify a Klaviyo ?",
        "Comment creer un flow de panier abandonne ?",
        "Comment importer des abonnes dans une liste ?",
        "Comment gerer les cles API de mon compte ?",
    ],
    "DE": [
        "Wie erstelle und sende ich eine E-Mail-Kampagne?",
        "Wie funktioniert die Klaviyo-Abrechnung?",
        "Wie richte ich SMS in Klaviyo ein?",
        "Wie funktionieren Anmeldeformulare?",
        "Wie verbinde ich Shopify mit Klaviyo?",
        "Wie erstelle ich einen Warenkorbabbruch-Flow?",
        "Wie importiere ich Abonnenten in eine Liste?",
        "Wie verwalte ich die API-Schlussel meines Kontos?",
    ],
    "PT": [
        "Como crio e envio uma campanha de email?",
        "Como funciona a cobranca da Klaviyo?",
        "Como configuro SMS na Klaviyo?",
        "Como funcionam os formularios de cadastro?",
        "Como conecto o Shopify a Klaviyo?",
        "Como crio um fluxo de carrinho abandonado?",
        "Como importo inscritos para uma lista?",
        "Como gerencio as chaves de API da minha conta?",
    ],
}

# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline(top_k=3)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_sources" not in st.session_state:
    st.session_state.show_sources = True

if "beginner_mode" not in st.session_state:
    st.session_state.beginner_mode = False

if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

if "user_input_main" not in st.session_state:
    st.session_state.user_input_main = ""

if "language_code" not in st.session_state:
    st.session_state.language_code = "EN"

if "pending_submit" not in st.session_state:
    st.session_state.pending_submit = False


def convert_markdown_links_to_html(text: str) -> str:
    """Convert markdown links in model output into clickable HTML anchors."""
    pattern = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
    return pattern.sub(
        r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
        text,
    )


def clean_assistant_content(text: str) -> str:
    """Strip legacy HTML wrappers that may have been stored in older chat messages."""
    cleaned = (text or "").strip()
    for _ in range(3):
        decoded = html.unescape(cleaned)
        if decoded == cleaned:
            break
        cleaned = decoded
    cleaned = re.sub(r"<strong>.*?</strong><br\s*/?>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"\bKlaviyo Assistant:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bAsistente de Klaviyo:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bAssistant Klaviyo:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bKlaviyo Assistent:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bAssistente Klaviyo:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</div\s*>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<div[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def append_jsonl(filename: str, payload: dict) -> None:
    """Persist lightweight analytics for common questions and failures."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with (LOG_DIR / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


def source_domain(url: str) -> str:
    """Return a short display domain for source links."""
    if not url:
        return "Local docs"
    domain = urlparse(url).netloc.lower()
    return domain.removeprefix("www.")


def submit_question(question: str, language_code: str) -> None:
    """Handle a user question from the input box, a follow-up button, or a starter prompt."""
    cleaned_question = question.strip()
    if not cleaned_question:
        return

    copy = LANGUAGE_OPTIONS[language_code]
    prior_messages = list(st.session_state.messages)

    st.session_state.messages.append({
        "role": "user",
        "content": cleaned_question,
    })

    with st.spinner(f"🔍 {copy['searching']}"):
        try:
            result = st.session_state.rag_pipeline.answer_question(
                cleaned_question,
                verbose=False,
                response_language=copy["llm_name"],
                chat_history=prior_messages,
            )

            answer = clean_assistant_content(result["answer"])
            if st.session_state.beginner_mode:
                if not answer.lower().startswith(("here", "to", "first", "the step", "you")):
                    answer = f"📖 **{copy['step_prefix']}**\n\n{answer}"

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": result["sources"],
                "followups": result["followup_questions"],
                "language_code": language_code,
                "retrieval": result["retrieval"],
            })

            append_jsonl("query_history.jsonl", {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "query": cleaned_question,
                "language_code": language_code,
                "source_count": len(result["sources"]),
                "weak_retrieval": result["retrieval"]["weak_retrieval"],
                "best_distance": result["retrieval"]["best_distance"],
                "average_distance": result["retrieval"]["average_distance"],
            })

            st.session_state.pending_question = ""
            st.session_state.pending_submit = False
            st.rerun()

        except Exception as exc:
            append_jsonl("error_history.jsonl", {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "query": cleaned_question,
                "language_code": language_code,
                "error": str(exc),
            })
            st.error(f"❌ An error occurred: {str(exc)}")
            st.info("Make sure you've run `pip install -r requirements.txt` and completed Step 3 (embeddings pipeline).")


current_language = st.session_state.language_code
copy = LANGUAGE_OPTIONS[current_language]

picker_col, _ = st.columns([1.4, 6.6])
with picker_col:
    selected_language = st.selectbox(
        copy["language_label"],
        options=list(LANGUAGE_OPTIONS.keys()),
        format_func=lambda code: f"{code} - {LANGUAGE_OPTIONS[code]['name']}",
        index=list(LANGUAGE_OPTIONS.keys()).index(current_language),
        key="language_selector",
    )
    if selected_language != current_language:
        st.session_state.language_code = selected_language
        st.rerun()

# Header
st.markdown(f"""
<div class="header">
    <h1>🤖 {copy["header_title"]}</h1>
    <p>{copy["header_subtitle"]}</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header(f"⚙️ {copy['settings']}")
    
    st.session_state.beginner_mode = st.toggle(
        copy["beginner_mode"],
        value=st.session_state.beginner_mode,
        help=copy["beginner_help"]
    )
    
    st.session_state.show_sources = st.toggle(
        copy["show_sources"],
        value=st.session_state.show_sources,
        help=copy["show_sources_help"]
    )
    
    top_k = st.slider(
        copy["num_sources"],
        min_value=1,
        max_value=10,
        value=3,
        help=copy["num_sources_help"]
    )
    
    if top_k != st.session_state.rag_pipeline.top_k:
        st.session_state.rag_pipeline.top_k = top_k
    
    st.divider()
    
    if st.button(f"🗑️ {copy['clear_history']}"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    example_markdown = "\n".join(f"- {question}" for question in EXAMPLE_QUESTIONS[current_language])
    st.markdown(f"### {copy['example_questions']}\n{example_markdown}")

# Main chat interface
st.markdown(f"### 💬 {copy['chat_title']}")

if not st.session_state.messages:
    st.markdown(
        f"""
        <div class="starter-card">
            <h3>{copy["starter_title"]}</h3>
            <p>{copy["starter_body"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    starter_cols = st.columns(2)
    for index, prompt in enumerate(EXAMPLE_QUESTIONS[current_language][:4]):
        with starter_cols[index % 2]:
            if st.button(prompt, key=f"starter_{current_language}_{index}", use_container_width=True):
                st.session_state.pending_question = prompt
                st.session_state.pending_submit = True

# Display chat history
for message_index, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f"""
        <div class="message-container user-message">
            <strong>{copy['you']}:</strong> {message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        assistant_content = clean_assistant_content(message["content"])
        response_language = message.get("language_code", current_language)
        retrieval = message.get("retrieval", {})
        st.markdown(
            f'<div class="assistant-meta">🌐 {response_language} {copy["answer_language_badge"]}</div>',
            unsafe_allow_html=True,
        )
        if retrieval.get("weak_retrieval"):
            st.markdown(
                f'<div class="retrieval-warning">{copy["limited_answer_warning"]}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(f"**{copy['assistant']}:**")
        st.markdown(assistant_content)
        
        # Show sources if available
        if st.session_state.show_sources and "sources" in message:
            with st.expander(f"📚 {copy['sources']}", expanded=False):
                for i, source in enumerate(message['sources'], 1):
                    source_title = source["source"]
                    source_url = source.get("url", "")
                    meta_parts = [source.get("category", "General"), source_domain(source_url)]
                    if source.get("locale"):
                        meta_parts.append(source["locale"].upper())
                    if source_url:
                        source_header = f'<a href="{source_url}" target="_blank" rel="noopener noreferrer">{source_title}</a>'
                    else:
                        source_header = source_title
                    st.markdown(f"""
                    <div class="source-badge">
                        <strong>{copy['source_label']} {i}:</strong> {source_header}<br>
                        <span class="source-meta">{' • '.join(meta_parts)}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show follow-up questions if available
        if "followups" in message and message['followups']:
            st.markdown(f"**💡 {copy['followups']}**")
            cols = st.columns(len(message['followups']))
            for col, followup in zip(cols, message['followups']):
                with col:
                    if st.button(followup, key=f"followup_{message_index}_{followup}", use_container_width=True):
                        st.session_state.pending_question = followup
                        st.session_state.pending_submit = True

# Input area
st.divider()

st.markdown(f"**{copy['ask_prompt']}**")

pending_question_value = st.session_state.pending_question
if pending_question_value and not st.session_state.pending_submit:
    st.session_state.user_input_main = pending_question_value
    st.session_state.pending_question = ""

with st.form("chat_input_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])

    with col1:
        user_input = st.text_input(
            copy["ask_prompt"],
            placeholder=copy["ask_placeholder"],
            key="user_input_main",
            label_visibility="collapsed"
        )

    with col2:
        send_button = st.form_submit_button(
            copy["send"],
            use_container_width=True,
            type="primary"
        )

if send_button and user_input.strip():
    submit_question(user_input, current_language)
elif st.session_state.pending_submit and pending_question_value:
    submit_question(pending_question_value, current_language)

# Footer
st.divider()
st.markdown(f"""
<div style='text-align: center; color: #999; font-size: 0.9rem;'>
    <p>{copy["footer"]}</p>
</div>
""", unsafe_allow_html=True)
