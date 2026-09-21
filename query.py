import os
import sys
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone
from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Check Streamlit Cloud st.secrets
try:
    import streamlit as _st
    if hasattr(_st, "secrets"):
        for _k in ["GEMINI_API_KEY", "GROQ_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_HOST"]:
            if _k in _st.secrets and not os.getenv(_k):
                os.environ[_k] = str(_st.secrets[_k])
except Exception:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
index = pc.Index(host=PINECONE_INDEX_HOST) if (pc and PINECONE_INDEX_HOST) else None

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

EMBEDDING_MODELS = [
    "gemini-embedding-001",
    "gemini-embedding-2",
    "gemini-embedding-2-preview"
]

GEMINI_GENERATIVE_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite"
]

GROQ_GENERATIVE_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "groq/compound",
    "groq/compound-mini"
]


def embed_query_with_retry(query: str, max_retries: int = 4) -> list[float]:
    if not genai_client:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
        
    for model_name in EMBEDDING_MODELS:
        for attempt in range(max_retries):
            try:
                response = genai_client.models.embed_content(
                    model=model_name,
                    contents=query,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY",
                        output_dimensionality=768
                    )
                )
                return response.embeddings[0].values
            except Exception as e:
                err_str = str(e)
                wait_time = 3.0 * (attempt + 1) if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) else 1.5 * (attempt + 1)
                logger.warning(f"Query embed attempt {attempt + 1} with {model_name} failed: {err_str[:60]}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                
    raise RuntimeError("Critical: Unable to embed query across all models.")


def retrieve_context_with_retry(query_vector: list[float], top_k: int = 3, max_retries: int = 3) -> List[Dict[str, Any]]:
    for attempt in range(max_retries):
        try:
            result = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            return result.get("matches", [])
        except Exception as e:
            wait_time = 2 ** attempt
            logger.warning(f"Pinecone query attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            
    logger.error("Pinecone search failed after max retries.")
    return []


def generate_with_groq(prompt: str, system_instruction: str, history: Optional[List[Dict[str, str]]] = None, model_name: str = "qwen/qwen3.8-27b") -> Optional[str]:
    chunks = list(generate_with_groq_stream(prompt, system_instruction, history, model_name))
    return "".join(chunks) if chunks else None


def generate_with_gemini(prompt: str, system_instruction: str, history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
    chunks = list(generate_with_gemini_stream(prompt, system_instruction, history))
    return "".join(chunks) if chunks else None


def generate_with_groq_stream(prompt: str, system_instruction: str, history: Optional[List[Dict[str, str]]] = None, model_name: str = "qwen/qwen3.8-27b"):
    if not GROQ_API_KEY:
        return
    
    client = Groq(api_key=GROQ_API_KEY)
    messages = [{"role": "system", "content": system_instruction}]
    if history:
        for h in history[-6:]:  # include last 3 conversational turns
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": prompt})
    
    for model in [model_name] + [m for m in GROQ_GENERATIVE_MODELS if m != model_name]:
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=1536,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content if chunk.choices else None
                if content:
                    yield content
            return
        except Exception as e:
            logger.warning(f"Groq stream failed with {model}: {e}. Falling back...")


def generate_with_gemini_stream(prompt: str, system_instruction: str, history: Optional[List[Dict[str, str]]] = None):
    contents = []
    if history:
        for h in history[-6:]:
            contents.append(f"{h['role'].upper()}: {h['content']}")
    contents.append(prompt)
    full_prompt = "\n\n".join(contents)

    for model_name in GEMINI_GENERATIVE_MODELS:
        try:
            response = genai_client.models.generate_content_stream(
                model=model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2
                )
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            return
        except Exception as e:
            logger.warning(f"Gemini stream failed with {model_name}: {e}. Retrying fallback...")


def generate_rag_answer_stream(
    query: str,
    top_k: int = 5,
    provider: str = "auto",
    history: Optional[List[Dict[str, str]]] = None,
    similarity_threshold: float = 0.50
):
    """
    Generate a streaming RAG answer with conversational memory and dynamic similarity filtering.
    Yields: (token_chunk, metadata_dict_or_None)
    """
    start_time = time.time()
    logger.info(f"Generating embedding for query: '{query}'...")
    try:
        query_vector = embed_query_with_retry(query)
    except Exception as e:
        yield (f"⚠️ System Error: Embedding service unavailable ({e})", {"matches": [], "engine": "Error", "latency": 0.0})
        return

    logger.info(f"Querying Pinecone with top_k={top_k}...")
    raw_matches = retrieve_context_with_retry(query_vector, top_k=top_k)
    
    # Filter matches above the relaxed 0.50 threshold
    matches = [m for m in raw_matches if m.get("score", 0.0) >= similarity_threshold]
    if not matches and raw_matches:
        # Fallback to top matches if scores hover slightly under
        matches = raw_matches[:2]

    context_blocks = []
    for idx, match in enumerate(matches, start=1):
        score = match.get("score", 0.0)
        metadata = match.get("metadata", {})
        source = metadata.get("source", "unknown")
        text = metadata.get("text", "")
        context_blocks.append(f"[[SOURCE: {source} | SIMILARITY: {score:.4f} | MATCH: {idx}]]\n{text}")

    context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant document excerpts found."

    # Human-Like Conversational Tone & Context Synthesis Prompt
    system_instruction = (
        "You are Chat_Doc, an intelligent and friendly AI document assistant.\n\n"
        "Your task is to answer user questions in a warm, conversational, human tone using the provided document context.\n\n"
        "GUIDELINES:\n"
        "1. Tone: Speak like a helpful expert colleague. Avoid mechanical disclaimers or rigid robotic statements.\n"
        "2. Context Synthesis: Use the CONTEXT CHUNKS below as your primary source. If a user asks about a specific file or topic, summarize the available information or related concepts found across the retrieved chunks.\n"
        "3. Graceful Fallbacks: If the retrieved chunks genuinely contain no relevant information about the query, respond naturally (e.g., \"I checked your indexed documents, but couldn't find specific details on that topic. Would you like me to summarize one of your other uploaded files instead?\")."
    )

    prompt = f"""CONTEXT CHUNKS:
{context_str}

USER QUESTION:
{query}
"""

    engine_used = "Gemini 2.0 Flash"
    stream_generator = None

    if provider == "groq" and GROQ_API_KEY:
        engine_used = "Groq LPU (Qwen 3.8 / GPT-OSS)"
        stream_generator = generate_with_groq_stream(prompt, system_instruction, history)
    elif provider == "gemini" and GEMINI_API_KEY:
        engine_used = "Gemini 2.0 Flash"
        stream_generator = generate_with_gemini_stream(prompt, system_instruction, history)
    else:  # Auto-routing
        if len(query.strip()) < 120 and GROQ_API_KEY:
            engine_used = "Groq LPU (Auto-Routed)"
            stream_generator = generate_with_groq_stream(prompt, system_instruction, history)
        else:
            engine_used = "Gemini 2.0 Flash (Auto-Routed)"
            stream_generator = generate_with_gemini_stream(prompt, system_instruction, history)

    if stream_generator is None and GEMINI_API_KEY:
        stream_generator = generate_with_gemini_stream(prompt, system_instruction, history)

    has_content = False
    if stream_generator:
        for token in stream_generator:
            if token:
                has_content = True
                yield (token, None)

    if not has_content:
        yield ("I could not retrieve sufficient details from the current context to answer your question completely.", None)

    elapsed = time.time() - start_time
    yield ("", {"matches": matches, "engine": engine_used, "latency": elapsed})


def generate_rag_answer(
    query: str, 
    top_k: int = 5, 
    return_matches: bool = False, 
    provider: str = "auto",
    history: Optional[List[Dict[str, str]]] = None
):
    """Synchronous wrapper for RAG answer generation."""
    chunks = []
    meta = {}
    for text_chunk, metadata in generate_rag_answer_stream(query, top_k=top_k, provider=provider, history=history):
        if text_chunk:
            chunks.append(text_chunk)
        if metadata:
            meta = metadata
            
    full_answer = "".join(chunks)
    matches = meta.get("matches", [])
    engine_used = meta.get("engine", "Auto")
    elapsed = meta.get("latency", 0.0)
    
    return (full_answer, matches, engine_used, elapsed) if return_matches else full_answer


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        answer = generate_rag_answer(query)
        print("\n================ Answer ================")
        print(answer)
        print("========================================")
    else:
        print("\n=== Interactive RAG Terminal (Type 'exit' to quit) ===")
        while True:
            try:
                user_input = input("\nEnter your question: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    break
                answer = generate_rag_answer(user_input)
                print("\n================ Answer ================")
                print(answer)
                print("========================================")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
