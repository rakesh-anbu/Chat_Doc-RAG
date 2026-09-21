import os
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone

from ingest import ingest_file, extract_text_universal
from query import generate_rag_answer, generate_rag_answer_stream, generate_with_gemini, generate_with_groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [
    "pdf", "json", "txt", "md", "csv",
    "docx", "doc", "dot", "dotx",
    "pptx", "ppt",
    "xlsx", "xls",
    "yaml", "yml", "xml",
    "epub",
    "rtf", "html", "htm",
    "hwpx", "hwp"
]

load_dotenv()

# Seamlessly synchronize Streamlit Cloud secrets into os.environ if running on cloud
try:
    if hasattr(st, "secrets"):
        for key in ["GEMINI_API_KEY", "GROQ_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_HOST"]:
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = str(st.secrets[key])
except Exception:
    pass

# Initialize Pinecone & check API statuses silently
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
index = pc.Index(host=PINECONE_INDEX_HOST) if (pc and PINECONE_INDEX_HOST) else None

# Sync state with filesystem on launch
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
existing_disk_files = list(data_dir.glob("*.*"))

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = {f.name for f in existing_disk_files}
if "active_docs_count" not in st.session_state:
    st.session_state.active_docs_count = len(existing_disk_files)
if "dynamic_suggestions" not in st.session_state:
    st.session_state.dynamic_suggestions = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

st.set_page_config(
    page_title="Chat_Doc",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Graceful API Key Missing Guard
if not PINECONE_API_KEY or not PINECONE_INDEX_HOST or not GEMINI_API_KEY:
    st.warning("⚠️ **Missing Configuration**: Ensure `GEMINI_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX_HOST` are set in your `.env` file to enable vector indexing and neural Q&A.")

# ==============================================================================
# 1. GLOBAL LAYOUT CSS SNIPPET & MODERN STYLING
# ==============================================================================
st.markdown("""
<style>
    /* 1. Structural Layout Containers */
    .main .block-container {
        max-width: 1150px !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
    }

    /* 2. Targeted Loading Status Suppression */
    div[data-testid="stStatusWidget"],
    .stStatusWidget {
        visibility: hidden !important;
        opacity: 0 !important;
    }

    .stAppDeployButton {
        display: none !important;
    }

    header[data-testid="stHeader"],
    [data-testid="stHeader"] {
        background: rgba(5, 13, 9, 0.75) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid rgba(52, 211, 153, 0.2) !important;
        height: 48px !important;
    }

    /* 3. Sidebar Container & Toggle */
    section[data-testid="stSidebar"][aria-expanded="true"] {
        z-index: 99999 !important;
        min-width: 300px !important;
        max-width: 320px !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0px !important;
        width: 0px !important;
    }

    button[data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button {
        z-index: 1000001 !important;
        pointer-events: auto !important;
        visibility: visible !important;
        display: inline-flex !important;
        background: rgba(16, 185, 129, 0.15) !important;
        border: 1px solid rgba(52, 211, 153, 0.4) !important;
        border-radius: 8px !important;
        color: #a7f3d0 !important;
        transition: all 0.2s ease !important;
    }
    button[data-testid="stSidebarCollapseButton"]:hover {
        background: rgba(16, 185, 129, 0.3) !important;
        box-shadow: 0 0 12px rgba(52, 211, 153, 0.6) !important;
    }

    button[data-testid="baseButton-header"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        z-index: 1000000 !important;
        pointer-events: auto !important;
        visibility: visible !important;
        display: inline-flex !important;
    }

    /* 4. Suggestion Pills Multi-Line Wrapping & Smart Adaptive Height */
    div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: stretch !important;
        gap: 12px !important;
    }

    div[data-testid="column"] {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: stretch !important;
    }

    div[data-testid="column"] > div {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }

    div[data-testid="column"] div[data-testid="stButton"] {
        height: 100% !important;
        display: flex !important;
        width: 100% !important;
    }

    /* Target all buttons to wrap text properly by default unless specifically overridden */
    div[data-testid="stButton"] > button {
        width: 100% !important;
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
        word-wrap: break-word !important;
        text-align: left !important;
        height: 100% !important;
        min-height: 54px !important;
        padding: 0.65rem 0.85rem !important;
        line-height: 1.35 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        background: rgba(13, 33, 24, 0.75) !important;
        border: 1px solid rgba(52, 211, 153, 0.35) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stButton"] > button p,
    div[data-testid="stButton"] > button span,
    div[data-testid="stButton"] > button div {
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
        text-align: left !important;
        font-size: 13px !important;
        line-height: 1.35 !important;
        display: block !important;
        width: 100% !important;
    }

    div[data-testid="stButton"] > button:hover {
        background: rgba(16, 185, 129, 0.22) !important;
        border-color: rgba(52, 211, 153, 0.8) !important;
        box-shadow: 0 0 16px rgba(52, 211, 153, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    /* Delete / Reset Specific Action Buttons within Document Card */
    div[data-testid="stExpander"] div[data-testid="stButton"] > button {
        min-height: 38px !important;
        height: auto !important;
        text-align: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        padding: 0.35rem 0.75rem !important;
    }
    div[data-testid="stExpander"] div[data-testid="stButton"] > button p {
        text-align: center !important;
        white-space: nowrap !important;
    }

    /* 6. Document Card Column Alignment & File Uploader */
    [data-testid="stFileUploader"] ul,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFilesContainer"] {
        max-height: 120px !important;
        overflow-y: auto !important;
    }

    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-bottom: 4rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    .stAppDeployButton {
        display: none !important;
    }
    footer {
        visibility: hidden;
    }
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    /* Living Fluid Ambient Background Gradient Animation */
    .stApp {
        background: radial-gradient(circle at 20% 20%, rgba(6, 78, 59, 0.18) 0%, transparent 45%),
                    radial-gradient(circle at 80% 80%, rgba(4, 120, 87, 0.15) 0%, transparent 50%),
                    radial-gradient(circle at 50% 50%, rgba(2, 44, 34, 0.25) 0%, transparent 70%),
                    #050d09 !important;
        background-attachment: fixed !important;
        animation: ambient-drift 20s ease-in-out infinite alternate !important;
    }
    
    @keyframes ambient-drift {
        0% { background-position: 0% 0%, 100% 100%, 50% 50%; }
        50% { background-position: 10% 20%, 90% 70%, 40% 60%; }
        100% { background-position: 0% 0%, 100% 100%, 50% 50%; }
    }
    
    /* Clean Hero Container */
    .hero-container {
        display: block !important;
        width: 100% !important;
        padding: 16px 20px !important;
        background: rgba(10, 26, 18, 0.6) !important;
        border: 1px solid rgba(52, 211, 153, 0.25) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(16px) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
        margin-bottom: 20px !important;
    }
    
    .hero-title {
        font-size: 24px !important;
        font-weight: 800 !important;
        color: #f0fdf4 !important;
        margin: 0 0 4px 0 !important;
        letter-spacing: -0.3px;
        text-shadow: 0 0 16px rgba(52, 211, 153, 0.4);
    }
    
    .hero-subtitle {
        font-size: 13.5px !important;
        color: #a7f3d0 !important;
        margin: 0 !important;
        font-weight: 500;
        opacity: 0.9;
    }
    
    /* Status Badges */
    .badge {
        padding: 5px 12px !important;
        border-radius: 8px !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        display: inline-block !important;
        margin-bottom: 6px !important;
        width: 100% !important;
        text-align: center !important;
    }
    
    .b-blue { background: rgba(30, 58, 138, 0.5) !important; color: #93c5fd !important; border: 1px solid rgba(147, 197, 253, 0.3) !important; }
    .b-purple { background: rgba(88, 28, 135, 0.5) !important; color: #d8b4fe !important; border: 1px solid rgba(216, 180, 254, 0.3) !important; }
    .b-green { background: rgba(6, 95, 70, 0.5) !important; color: #6ee7b7 !important; border: 1px solid rgba(110, 231, 183, 0.35) !important; }
    .b-amber { background: rgba(146, 64, 14, 0.5) !important; color: #fde68a !important; border: 1px solid rgba(253, 230, 138, 0.3) !important; }

    /* Glassmorphic Document Pill */
    .doc-pill {
        background: rgba(13, 33, 24, 0.6) !important;
        border: 1px solid rgba(52, 211, 153, 0.25) !important;
        border-radius: 10px !important;
        padding: 8px 12px !important;
        font-size: 13px !important;
        color: #f1f5f9 !important;
        font-weight: 500 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        margin-bottom: 6px !important;
        backdrop-filter: blur(12px) !important;
    }
    
    .doc-name {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 70%;
    }
    
    .doc-size {
        font-size: 11px;
        color: #6ee7b7;
        font-weight: 600;
        margin-left: 8px;
    }

    /* Natural Fluid File Uploader */
    [data-testid="stFileUploader"] section {
        padding: 14px !important;
        border-radius: 12px !important;
        background: rgba(10, 28, 20, 0.5) !important;
        border: 1px dashed rgba(52, 211, 153, 0.35) !important;
    }
    
    [data-testid="stFileUploader"] li {
        margin-bottom: 6px !important;
        background: rgba(16, 185, 129, 0.12) !important;
        border: 1px solid rgba(52, 211, 153, 0.25) !important;
        border-radius: 8px !important;
        padding: 6px 10px !important;
        color: #f1f5f9 !important;
    }
    
    /* Clean Empty State Placeholder */
    .empty-chat-box {
        text-align: center;
        padding: 36px 20px;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 16px;
        background: rgba(10, 26, 18, 0.5);
        backdrop-filter: blur(16px);
        margin: 16px 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    .pulsing-orb {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #a7f3d0, #10b981 40%, #064e3b 80%, #022c22);
        box-shadow: 0 0 35px rgba(52, 211, 153, 0.7), inset 0 0 16px rgba(255, 255, 255, 0.5);
        animation: orb-pulse 4s infinite ease-in-out;
        position: relative;
        margin-bottom: 14px;
    }
    
    .orb-ring {
        position: absolute;
        top: -10px;
        left: -10px;
        right: -10px;
        bottom: -10px;
        border: 1.5px dashed rgba(52, 211, 153, 0.5);
        border-radius: 50%;
        animation: ring-rotate 16s linear infinite;
    }
    
    @keyframes orb-pulse {
        0%, 100% { transform: scale(1); box-shadow: 0 0 30px rgba(52, 211, 153, 0.6); }
        50% { transform: scale(1.08); box-shadow: 0 0 50px rgba(52, 211, 153, 0.9); }
    }
    
    @keyframes ring-rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    /* Futuristic Cybernetic Holographic Loading Card */
    .cyber-loader-card {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.45) 0%, rgba(10, 30, 22, 0.92) 100%) !important;
        border: 1px solid rgba(52, 211, 153, 0.55) !important;
        border-radius: 14px !important;
        padding: 20px 24px !important;
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.3), inset 0 0 20px rgba(52, 211, 153, 0.15) !important;
        backdrop-filter: blur(20px) !important;
        position: relative !important;
        overflow: hidden !important;
        margin: 12px 0 !important;
    }
    
    .cyber-loader-card::before {
        content: "" !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(52, 211, 153, 0.25), transparent) !important;
        animation: cyber-scan 2.2s infinite linear !important;
    }
    
    .cyber-loader-inner {
        display: flex !important;
        align-items: center !important;
        gap: 16px !important;
        position: relative !important;
        z-index: 2 !important;
    }
    
    /* 3D Cyber Hexagon / Ring Spinner */
    .cyber-ring-spinner {
        width: 36px !important;
        height: 36px !important;
        border-radius: 50% !important;
        border: 3px solid rgba(52, 211, 153, 0.2) !important;
        border-top-color: #10b981 !important;
        border-right-color: #34d399 !important;
        border-left-color: #059669 !important;
        box-shadow: 0 0 16px rgba(52, 211, 153, 0.9), inset 0 0 8px rgba(52, 211, 153, 0.6) !important;
        animation: cyber-spin 0.75s cubic-bezier(0.68, -0.55, 0.27, 1.55) infinite !important;
        flex-shrink: 0 !important;
    }
    
    .cyber-loader-text {
        color: #a7f3d0 !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        letter-spacing: 0.4px !important;
        text-shadow: 0 0 10px rgba(52, 211, 153, 0.8) !important;
    }
    
    .cyber-loader-sub {
        font-size: 11px !important;
        color: #6ee7b7 !important;
        opacity: 0.8 !important;
        margin-top: 2px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Cyber Laser Progress Bar */
    .cyber-progress-track {
        width: 100% !important;
        height: 10px !important;
        background: rgba(10, 30, 22, 0.8) !important;
        border: 1px solid rgba(52, 211, 153, 0.4) !important;
        border-radius: 6px !important;
        overflow: hidden !important;
        margin-top: 10px !important;
        position: relative !important;
    }
    
    .cyber-progress-fill {
        height: 100% !important;
        background: linear-gradient(90deg, #047857 0%, #10b981 40%, #34d399 75%, #6ee7b7 100%) !important;
        background-size: 200% 100% !important;
        animation: laser-sweep 1.8s linear infinite !important;
        box-shadow: 0 0 16px rgba(52, 211, 153, 0.9) !important;
        border-radius: 4px !important;
        transition: width 0.3s ease !important;
    }

    /* Hide native running status animation in top-right header & default spinner */
    [data-testid="stStatusWidget"],
    [data-testid="stStatusWidget"] *,
    div[data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
    }
    div[data-testid="stSpinner"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Cyber Futuristic Loader & Radial Gradient Spinner Styles */
    .cyber-spinner {
        width: 24px;
        height: 24px;
        border: 3px solid rgba(0, 242, 254, 0.2);
        border-top: 3px solid #00F2FE;
        border-right: 3px solid #7928CA;
        border-radius: 50%;
        animation: cyber-spin-fast 0.8s linear infinite;
        box-shadow: 0 0 12px rgba(0, 242, 254, 0.5);
    }
    @keyframes cyber-spin-fast {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Cyber Pulse & Radial Glow Effects */
    .cyber-pulse {
        width: 18px !important;
        height: 18px !important;
        background-color: #00F2FE !important;
        border-radius: 50% !important;
        box-shadow: 0 0 12px #00F2FE, inset 0 0 4px #ffffff !important;
        animation: cyber-pulse-anim 1s infinite alternate ease-in-out !important;
        flex-shrink: 0 !important;
    }
    @keyframes cyber-pulse-anim {
        0% { transform: scale(0.85); opacity: 0.55; box-shadow: 0 0 6px #00F2FE; }
        100% { transform: scale(1.2); opacity: 1; box-shadow: 0 0 18px #00F2FE, 0 0 30px rgba(0, 242, 254, 0.4); }
    }
</style>
""", unsafe_allow_html=True)


def cyber_loader(message="PROCESSING QUANTUM INDEX..."):
    """Futuristic Glassmorphic Loader Component returning an active Streamlit empty slot."""
    slot = st.empty()
    slot.markdown(f"""
        <div style="
            display: flex; 
            align-items: center; 
            justify-content: center;
            gap: 16px; 
            padding: 1rem 1.5rem; 
            background: rgba(15, 23, 42, 0.85); 
            backdrop-filter: blur(12px);
            border: 1px solid rgba(0, 242, 254, 0.4); 
            border-radius: 12px; 
            box-shadow: 0 0 20px rgba(0, 242, 254, 0.2);
            margin: 1.5rem 0;
        ">
            <div class="neon-ring"></div>
            <span style="
                color: #00F2FE; 
                font-family: 'Courier New', monospace; 
                font-weight: 700; 
                letter-spacing: 2px;
                text-shadow: 0 0 8px rgba(0, 242, 254, 0.6);
            ">
                {message}
            </span>
        </div>
        <style>
            .neon-ring {{
                width: 20px;
                height: 20px;
                border: 3px solid rgba(0, 242, 254, 0.2);
                border-top: 3px solid #00F2FE;
                border-right: 3px solid #7928CA;
                border-radius: 50%;
                animation: spin-ring 0.7s linear infinite;
            }}
            @keyframes spin-ring {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
    """, unsafe_allow_html=True)
    return slot


def show_futuristic_loader(message="⚡ PROCESSING QUANTUM INDEX...", container=None):
    """Render a sleek cyberpunk / futuristic loading overlay with cyan glowing pulse."""
    if container is None:
        container = st.empty()
    return cyber_loader(message)

# Main Clean Hero Header (No badge clutter)
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Chat_Doc</div>
    <div class="hero-subtitle">Universal Multi-Format AI Document Intelligence &amp; Neural Q&amp;A</div>
</div>
""", unsafe_allow_html=True)


def format_size_mb(size_bytes: int) -> str:
    """Format file size uniformly in Megabytes (MB) to one decimal place."""
    size_mb = size_bytes / (1024 * 1024)
    return f"{size_mb:.1f} MB" if size_mb >= 0.1 else f"{max(0.1, size_mb):.1f} MB"


def delete_single_file(file_path: Path):
    try:
        filename = file_path.name
        file_path.unlink(missing_ok=True)
        st.session_state.processed_files.discard(filename)
        
        # Delete from Pinecone index by source metadata filter
        try:
            index.delete(filter={"source": {"$eq": filename}})
        except Exception as pe:
            logger.warning(f"Could not delete vectors for {filename} from Pinecone: {pe}")
            
        st.success(f"Removed '{filename}' and purged associated vector embeddings.")
        st.rerun()
    except Exception as e:
        st.error(f"Error removing file: {e}")


def generate_questions_from_document(file_path: str, filename: str) -> List[str]:
    """Dynamically generate 3 high-value technical questions based on document text context."""
    try:
        raw_text = extract_text_universal(file_path)
        if not raw_text or len(raw_text.strip()) < 50:
            return []
        
        # Take an informative sample of the document
        sample_context = raw_text[:4000].strip()
        prompt = (
            f"Analyze the following excerpt from the document '{filename}' and formulate exactly 3 concise, insightful, "
            f"and high-value sample questions that a user could ask to test knowledge retrieval from this document.\n\n"
            f"Document Sample:\n\"\"\"\n{sample_context}\n\"\"\"\n\n"
            f"Rules:\n"
            f"- Return ONLY the 3 questions, each on a new line.\n"
            f"- Do not include numbering, bullets, quotation marks, or markdown preamble/postamble.\n"
            f"- Questions must be under 90 characters each."
        )
        sys_inst = "You are an expert document intelligence assistant that generates concise, relevant questions from context."
        
        response_text = None
        # Try Gemini first for deep comprehension, fallback to Groq
        if GEMINI_API_KEY:
            response_text = generate_with_gemini(prompt, sys_inst)
        if not response_text and GROQ_API_KEY:
            response_text = generate_with_groq(prompt, sys_inst)
            
        if response_text:
            lines = [line.strip().lstrip("0123456789.-*• ") for line in response_text.strip().split("\n") if line.strip()]
            valid_questions = [q for q in lines if len(q) > 10]
            if len(valid_questions) >= 3:
                return valid_questions[:3]
    except Exception as e:
        logger.warning(f"Failed to dynamically generate questions for {filename}: {e}")
    return []


def wipe_all_knowledge():
    try:
        data_files = list(Path("data").glob("*.*"))
        for f in data_files:
            f.unlink(missing_ok=True)
        if index:
            index.delete(delete_all=True)
        st.session_state.processed_files.clear()
        st.session_state.messages = []
        st.session_state.dynamic_suggestions = []
        st.session_state.active_docs_count = 0
        st.success("All documents, vectors, and conversation history reset completely!")
        st.rerun()
    except Exception as e:
        st.error(f"Error resetting database: {e}")


def process_file_list(files_list):
    if not files_list:
        return
    
    new_files = [f for f in files_list if f.name not in st.session_state.processed_files]
    if not new_files:
        return
    
    total_files = len(new_files)
    status_container = st.empty()
    latest_generated_questions = []
    
    for idx, file_obj in enumerate(new_files, start=1):
        filename = file_obj.name
        save_path = data_dir / filename
        
        # Save locally
        with open(save_path, "wb") as f:
            f.write(file_obj.getbuffer())
            
        def update_file_progress(val):
            overall_pct = int((((idx - 1) + val) / total_files) * 100)
            status_container.markdown(f"""
            <div class="cyber-loader-card">
                <div class="cyber-loader-inner">
                    <div class="cyber-ring-spinner"></div>
                    <div style="flex: 1;">
                        <div class="cyber-loader-text">⚡ VECTORIZING DOCUMENT ({idx}/{total_files}): {filename}</div>
                        <div class="cyber-loader-sub">Pinecone 768-dim Embeddings &bull; {overall_pct}% Completed</div>
                    </div>
                </div>
                <div class="cyber-progress-track">
                    <div class="cyber-progress-fill" style="width: {overall_pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        try:
            start_t = time.time()
            ingest_file(str(save_path), batch_size=96, progress_callback=update_file_progress)
            elapsed = time.time() - start_t
            st.session_state.processed_files.add(filename)
            
            # Generate dynamic questions from context
            doc_questions = generate_questions_from_document(str(save_path), filename)
            if doc_questions:
                latest_generated_questions = doc_questions
        except Exception as e:
            st.error(f"Error indexing '{filename}': {e}")
            
    if latest_generated_questions:
        st.session_state.dynamic_suggestions = latest_generated_questions
        
    status_container.markdown(f"""
    <div class="cyber-loader-card" style="border-color: rgba(52, 211, 153, 0.9);">
        <div class="cyber-loader-inner">
            <span style="font-size: 24px;">✅</span>
            <div>
                <div class="cyber-loader-text" style="color: #6ee7b7;">INGESTION COMPLETE</div>
                <div class="cyber-loader-sub">Successfully indexed {total_files} document(s) into Pinecone Serverless.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(0.9)
    st.rerun()


# ==============================================================================
# 2. DOCUMENT MANAGEMENT CARD: TWO EQUAL HORIZONTAL COLUMNS
# ==============================================================================
with st.expander("📂 Document Ingestion & Knowledge Base Management", expanded=True):
    col_upload, col_list = st.columns([1, 1], gap="medium")
    
    with col_upload:
        st.markdown("**Drag & Drop Documents (Auto-Indexed)**")
        st.caption("Supported formats: PDF, JSON, Word, PowerPoint, Excel, EPUB, YAML, XML, TXT, CSV, RTF, HTML, HWPX")
        uploaded_files = st.file_uploader(
            "Choose document(s)", 
            type=SUPPORTED_EXTENSIONS, 
            accept_multiple_files=True, 
            key="main_uploader",
            label_visibility="collapsed",
            help="Drag and drop or browse files to auto-index into Pinecone 768-dim vector space."
        )
        
        if uploaded_files:
            process_file_list(uploaded_files)
                        
    with col_list:
        st.markdown("**Active Indexed Documents**")
        current_data_files = list(Path("data").glob("*.*"))
        if current_data_files:
            for f in current_data_files:
                size_str = format_size_mb(f.stat().st_size)
                c_name, c_del = st.columns([3.0, 1.0])
                with c_name:
                    st.markdown(
                        f'<div class="doc-pill" title="{f.name}"><span class="doc-name"><b>{f.name}</b></span><span class="doc-size">{size_str}</span></div>', 
                        unsafe_allow_html=True
                    )
                with c_del:
                    if st.button("Delete", key=f"del_{f.name}", type="secondary", use_container_width=True, help=f"Remove {f.name} and purge its vectors from Pinecone"):
                        delete_single_file(f)
            st.divider()
            if st.button("Wipe All Documents & Reset Index", key="wipe_btn", type="secondary", use_container_width=True, help="Purge all documents and reset the entire Pinecone index"):
                wipe_all_knowledge()
        else:
            st.caption("No documents indexed yet. Upload documents on the left to begin.")


# ==============================================================================
# 3. SIDEBAR: AI ENGINE, SYSTEM METADATA STATUS EXPANDER & TELEMETRY
# ==============================================================================
with st.sidebar:
    st.markdown("### AI Generation Engine")
    provider_option = st.selectbox(
        "Select LLM Engine",
        [
            "⚡ Auto-Failover (Dual Engine)",
            "🚀 Groq LPU (Ultra Fast)",
            "✨ Google Gemini 2.0 Flash"
        ],
        index=0,
        help="Select which AI engine synthesizes grounded answers. Auto-Failover automatically routes between Groq and Gemini."
    )
    
    provider_map = {
        "⚡ Auto-Failover (Dual Engine)": "auto",
        "🚀 Groq LPU (Ultra Fast)": "groq",
        "✨ Google Gemini 2.0 Flash": "gemini"
    }
    selected_provider = provider_map[provider_option]

    st.markdown("---")
    
    # 🌐 Collapsible System Metadata & Status Expander
    with st.expander("🌐 System Metadata & Status", expanded=True):
        st.markdown('<div class="badge b-blue">Pinecone: rag-knowledge-base</div>', unsafe_allow_html=True)
        st.markdown('<div class="badge b-purple">Embedding: gemini-embedding-001 (768-dim)</div>', unsafe_allow_html=True)
        st.markdown('<div class="badge b-green"><span class="status-dot-live"></span>LLM: Gemini 2.0 Flash / Groq LPU</div>', unsafe_allow_html=True)
        st.markdown('<div class="badge b-amber">Cloud Spend: $0.00 (Free Tier)</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Search Parameters & API Status", expanded=False):
        st.markdown("**API Connections**")
        st.markdown("🟢 **Google Gemini API**: Connected" if GEMINI_API_KEY else "🔴 **Google Gemini**: Missing API Key")
        st.markdown("🟢 **Groq LPU Engine**: Connected" if GROQ_API_KEY else "🟡 **Groq LPU**: Not Configured")
        st.markdown("🟢 **Pinecone Serverless**: Connected" if PINECONE_API_KEY else "🔴 **Pinecone**: Missing API Key")
        
        st.markdown("---")
        top_k_val = st.slider("Pinecone Context Chunks (top_k)", min_value=1, max_value=10, value=5)
        st.caption("Controls the number of relevant passages retrieved per query.")

    st.markdown("---")
    st.markdown("### Knowledge Base Stats")
    # Synchronize active_docs_count in session state
    disk_count = len(list(Path("data").glob("*.*")))
    st.session_state["active_docs_count"] = max(disk_count, len(st.session_state.get("processed_files", set())))
    st.metric(label="Active Documents", value=st.session_state["active_docs_count"])
    st.caption("Vector Index: `rag-knowledge-base` (768-dim)")


# ==============================================================================
# 4. MAIN CHAT & CONVERSATIONAL Q&A CANVAS
# ==============================================================================
st.markdown("### Grounded Document Q&A")

# Clean Rounded Empty State Placeholder with Subtle Border Highlights
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-chat-box">
        <div class="pulsing-orb">
            <div class="orb-ring"></div>
        </div>
        <h4 style="margin: 0 0 6px 0; color: #f0fdf4; font-size: 16px; font-weight: 700;">No messages yet — Ask a question about your indexed documents below...</h4>
        <p style="font-size: 13px; margin: 0; color: #94a3b8; max-width: 550px;">
            Ask any question grounded against your indexed knowledge base, or click any of the suggested technical questions below.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "engine" in msg and msg.get("engine"):
            st.caption(f"⚡ *Synthesized via {msg['engine']} in {msg.get('latency', 0.0):.2f}s*")
        if "matches" in msg and msg["matches"]:
            with st.expander("View Retrieved Sources and Pinecone Similarity Scores"):
                for idx, match in enumerate(msg["matches"], start=1):
                    score = match.get("score", 0.0)
                    meta = match.get("metadata", {})
                    source = meta.get("source", "Unknown")
                    text = meta.get("text", "")
                    st.markdown(f"**Match {idx}** | Source: `{source}` | Similarity Score: `{score:.4f}`")
                    st.caption(text)
                    if idx < len(msg["matches"]):
                        st.divider()


# ==============================================================================
# 5. SUGGESTED QUESTIONS (FILE-AWARE & DYNAMICALLY GENERATED)
# ==============================================================================
indexed_docs = list(Path("data").glob("*.*"))
indexed_docs_count = len(indexed_docs)

if len(st.session_state.messages) < 2:
    st.markdown("### Suggested Questions")
    
    if indexed_docs_count == 0:
        suggestions = [
            "⚡ How does dual-engine failover operate?",
            "📄 Supported vector index dimensions?",
            "🔍 How to configure Pinecone top_k search?"
        ]
    else:
        # Check if we have dynamically generated LLM suggestions stored
        if st.session_state.get("dynamic_suggestions"):
            suggestions = st.session_state["dynamic_suggestions"][:3]
        else:
            first_name = indexed_docs[0].name
            suggestions = [
                f"Summarize key findings from {first_name}",
                "List core methodologies in the indexed files",
                "Extract primary conclusions & data points"
            ]

    # Render suggestion pills cleanly across equal 3 columns with gap
    st.markdown('<div class="suggestion-container">', unsafe_allow_html=True)
    cols = st.columns(3, gap="small")
    for idx, question in enumerate(suggestions[:3]):
        with cols[idx]:
            st.markdown('<div class="suggestion-btn">', unsafe_allow_html=True)
            if st.button(
                question, 
                key=f"sug_{idx}", 
                use_container_width=True,
                disabled=st.session_state.get("is_processing", False)
            ):
                st.session_state["user_prompt"] = question
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

prompt_to_process = None
if "user_prompt" in st.session_state and st.session_state["user_prompt"]:
    prompt_to_process = st.session_state.pop("user_prompt")
else:
    user_chat_input = st.chat_input("Ask any question grounded against your indexed documents...")
    if user_chat_input:
        prompt_to_process = user_chat_input

if prompt_to_process:
    # Append user turn
    st.session_state.messages.append({"role": "user", "content": prompt_to_process})
    with st.chat_message("user"):
        st.markdown(prompt_to_process)
        
    with st.chat_message("assistant"):
        loader_placeholder = st.empty()
        loader_placeholder.markdown("""
        <div class="cyber-loader-card">
            <div class="cyber-loader-inner">
                <div class="cyber-ring-spinner"></div>
                <div>
                    <div class="cyber-loader-text">⚡ SCANNING PINECONE 768-DIM VECTOR SPACE...</div>
                    <div class="cyber-loader-sub">Computing Cosine Similarity &bull; Grounding Dual-Engine LLM Context</div>
                </div>
            </div>
            <div class="cyber-progress-track">
                <div class="cyber-progress-fill" style="width: 100%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Build conversational history (excluding current user message)
        conversation_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
            if m.get("content")
        ]
        
        # Stream response chunks progressively
        collected_metadata = {}
        def stream_response_generator():
            for text_chunk, metadata in generate_rag_answer_stream(
                prompt_to_process,
                top_k=top_k_val if 'top_k_val' in locals() else 5,
                provider=selected_provider,
                history=conversation_history,
                similarity_threshold=0.50
            ):
                if metadata:
                    collected_metadata.update(metadata)
                if text_chunk:
                    yield text_chunk

        gen = stream_response_generator()
        # Peek / initialize generator to finish scanning before clearing the loader
        loader_placeholder.empty()
        
        full_answer = st.write_stream(gen)
        
        matches = collected_metadata.get("matches", [])
        engine_used = collected_metadata.get("engine", "Dual Engine")
        latency = collected_metadata.get("latency", 0.0)
        
        if "Failover" in engine_used:
            st.toast("⚠️ Engine busy or rate limited — Switched to backup model", icon="⚠️")
            
        # Dual-Engine Speed & Metadata Badge
        badge_icon = "⚡" if "Groq" in engine_used else "✨"
        word_count = len(str(full_answer).split())
        tps = int(word_count / max(0.1, latency) * 1.3)
        st.markdown(
            f'<div style="font-size: 11.5px; color: #6ee7b7; font-family: monospace; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(52, 211, 153, 0.25); border-radius: 6px; padding: 4px 10px; display: inline-block; margin-top: 4px;">'
            f'{badge_icon} <b>Model:</b> {engine_used} &nbsp;|&nbsp; <b>Latency:</b> {latency:.2f}s &nbsp;|&nbsp; <b>Speed:</b> ~{tps} tokens/s'
            f'</div>',
            unsafe_allow_html=True
        )
        
        if matches:
            with st.expander("View Retrieved Sources and Pinecone Similarity Scores"):
                for idx, match in enumerate(matches, start=1):
                    score = match.get("score", 0.0)
                    meta = match.get("metadata", {})
                    source = meta.get("source", "Unknown")
                    text = meta.get("text", "")
                    st.markdown(f"**Match {idx}** | Source: `{source}` | Similarity Score: `{score:.4f}`")
                    st.caption(text)
                    if idx < len(matches):
                        st.divider()
                    
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "matches": matches,
            "engine": engine_used,
            "latency": latency
        })
