
# Retrieval-Augmented Generation (RAG) Assignment Report

## 1. System Overview
This project implements an end-to-end, production-grade Retrieval-Augmented Generation (RAG) architecture. It enables grounded, hallucination-free question answering across multi-format local enterprise documents (.pdf and .json) using Pinecone Serverless and Google Gemini AI.

## 2. Tech Stack & Configuration
- **Vector Database**: Pinecone Serverless (Metric: cosine, Dimension: 768)
- **Embeddings Model**: Google Gemini API (gemini-embedding-001, 768-dim output)
- **Generation Model**: Google Gemini API (gemini-3.5-flash with auto-failover)
- **Ingestion**: Automated chunking (500 tokens, 50-token overlap) for PDF and JSON
- **Execution**: Autonomous Python terminal scripts + Postman REST API collection

## 3. Ingestion & Retrieval Verification
1. **JSON Ingestion**: Ingested sample_knowledge.json with company policies.
2. **PDF Ingestion**: Ingested company_handbook.pdf with employee operations guide.
3. **Guardrail Testing**: Out-of-domain queries successfully trigger strict 'Information not found in context' output.

## 4. Postman REST Specs
All four core endpoints were tested in Postman:
1. Gemini Embeddings API (POST gemini-embedding-001) -> 768-dim vector
2. Pinecone Upsert API (POST /vectors/upsert) -> upsertedCount: 1
3. Pinecone Query API (POST /query) -> top_k matches with cosine similarity scores
4. Gemini Generation API (POST gemini-3.5-flash) -> grounded answer synthesis

