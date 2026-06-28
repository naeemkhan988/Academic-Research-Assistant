---
title: Academic Research Assistant
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: docker
app_port: 9838
pinned: false
---

# 🔬 AI Academic Research Assistant

AI-powered Academic Research Assistant for literature review, research gap analysis, and citation generation. Searches **8 academic databases** in parallel and analyzes results using multi-AI providers.

## Features

- **Parallel Search** — Queries arXiv, PubMed, Semantic Scholar, CORE, CrossRef, OpenAlex, DBLP, and Europe PMC simultaneously (~5s)
- **Multi-AI Fallback** — Uses Gemini, Groq, and Ollama with automatic failover
- **Analysis Types** — Literature Summary, Research Gaps, Compare Studies, Methodologies
- **Auto Citations** — Generates citations in APA, MLA, Chicago, and IEEE formats
- **PDF Export** — Download formatted research reports as PDF
- **Hybrid RAG** — FAISS vector search + keyword matching for relevant paper retrieval

## Tech Stack

- **Backend**: Flask, Python 3.10
- **AI/ML**: Sentence Transformers, FAISS, Google Gemini, Groq
- **APIs**: 8 academic databases with async httpx
- **Frontend**: Bootstrap 5, Glassmorphism UI
