# rag_pipeline.py
"""
Enhanced RAG Pipeline with:
- Hybrid Search (Keyword + Semantic)
- FAISS Index Persistence
- Multi-Provider LLM Fallback (Gemini → Groq → Ollama)
- Streaming AI Responses
"""

import os
import json
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from collections import Counter
from config import Config
from database import PaperManager
from utils import TextProcessor
from typing import List, Dict, Any, Generator, Optional, Set
from llm_router import llm_router


def clean_report_text(text: str) -> str:
    """Thoroughly sanitize report text to ensure clean, human-friendly presentation with no broken symbols"""
    if not text:
        return ""
    # Fix common mojibake / encoding sequences
    replacements = {
        'â€“': ' ',
        'â€”': ' ',
        'â€™': "'",
        'â€˜': "'",
        'â€œ': '"',
        'â€': '"',
        'â': ' ',
        'NaÃ¯ve': 'Naive',
        'naÃ¯ve': 'naive',
        'Ã¯': 'i',
        'Ã©': 'e',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove non-printable / control characters (C0 & C1 controls, invisible unicode, missing glyph blocks, macrons)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f\u200b-\u200f\ufeff\ufffd\u2580-\u259f\u25a0-\u25ff\u2b1b-\u2b1f\u00af\u02c9\u0304]', ' ', text)

    # Remove hyphens inside compound words (e.g. data-driven -> data driven, Cross-lingual -> Cross lingual)
    text = re.sub(r'(\w+)[-\u2010-\u2015](\w+)', r'\1 \2', text)
    # Remove standalone hyphens, dashes, and vertical pipes
    text = re.sub(r'[-\u2010-\u2015|]+', ' ', text)

    # Strip markdown headers and bold asterisks
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'^\s*[\*\-]\s+', '', text, flags=re.MULTILINE)

    # Ensure all section titles are placed cleanly on their own line with proper spacing even if embedded mid-text
    headings = [
        'AI ANALYSIS',
        'LITERATURE SUMMARY',
        'RESEARCH GAPS',
        'COMPARISON OF STUDIES',
        'METHODOLOGIES',
        'APPLICATIONS AND IMPLICATIONS',
        'CHALLENGES AND LIMITATIONS',
        'FUTURE DIRECTIONS'
    ]
    for h in headings:
        pattern = re.compile(rf'(?:\s*|^)(?:\d+[\.\)]\s*)?(?:#+\s*)?(?:\*\*)?{h}(?::|\s*-\s*)?(?:\*\*)?(?=\s+[A-Z0-9]|$)', re.IGNORECASE)
        text = pattern.sub(f'\n\n{h}\n', text)

    # Normalize multiple spaces per line without stripping newlines
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


class HybridSearchEngine:
    """
    Hybrid Search combining:
    - Keyword-based scoring (BM25-like)
    - Semantic vector similarity (FAISS)
    """

    def __init__(self):
        self.keyword_weight = Config.KEYWORD_WEIGHT
        self.semantic_weight = Config.SEMANTIC_WEIGHT

    def calculate_keyword_score(self, query: str, text: str) -> float:
        """Calculate keyword relevance score (simplified BM25)"""
        if not text or not query:
            return 0.0

        query_terms = set(query.lower().split())
        text_lower = text.lower()

        # Term frequency scoring
        matches = sum(1 for term in query_terms if term in text_lower)

        # Exact phrase bonus
        if query.lower() in text_lower:
            matches += 2

        # Normalize by query length
        score = matches / (len(query_terms) + 1)
        return min(score, 1.0)

    def calculate_hybrid_score(self, semantic_score: float, keyword_score: float) -> float:
        """Combine semantic and keyword scores"""
        return (self.semantic_weight * semantic_score) + (self.keyword_weight * keyword_score)


class FAISSIndexManager:
    """
    Manages FAISS index persistence to avoid rebuilding on every query
    """

    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = index_path
        self.metadata_path = metadata_path
        # ANNOTATED: mypy cannot infer the type from a bare `None` literal
        self.index: Optional[Any] = None
        # ANNOTATED: mypy cannot infer the element type from an empty `set()`
        self.indexed_titles: Set[str] = set()
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata about indexed papers"""
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.indexed_titles = set(data.get('titles', []))
        except Exception as e:
            print(f"Error loading FAISS metadata: {e}")
            self.indexed_titles = set()

    def _save_metadata(self):
        """Save metadata about indexed papers"""
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'titles': list(self.indexed_titles),
                    'count': len(self.indexed_titles)
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving FAISS metadata: {e}")

    def load_index(self) -> Optional[Any]:
        """Load existing FAISS index"""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                return self.index
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
        return None

    def save_index(self):
        """Save FAISS index to disk"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
                self._save_metadata()
        except Exception as e:
            print(f"Error saving FAISS index: {e}")

    def is_paper_indexed(self, title: str) -> bool:
        """Check if a paper is already indexed"""
        return title.lower().strip() in self.indexed_titles

    def mark_as_indexed(self, title: str):
        """Mark a paper as indexed"""
        self.indexed_titles.add(title.lower().strip())

    def get_new_papers(self, papers: List[Dict]) -> List[Dict]:
        """Filter out already indexed papers"""
        return [p for p in papers if not self.is_paper_indexed(p.get('title', ''))]


class RAGPipeline:
    """
    Enhanced RAG Pipeline with:
    - Hybrid Search (Keyword + Semantic)
    - FAISS Persistence
    - Multi-Provider LLM Fallback (Gemini → Groq → Ollama)
    - Streaming Support
    """

    def __init__(self):
        # LLM Router handles all AI generation with automatic provider fallback
        self.router = llm_router
        print("[RAG] Pipeline using Multi-Provider LLM Router")

        # Show provider status
        status = self.router.get_status()
        for pname, pinfo in status['providers'].items():
            icon = '[OK]' if pinfo['available'] else '[--]'
            print(f"   {icon} {pname}: model={pinfo['model']}, available={pinfo['available']}")

        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.paper_manager = PaperManager()
        self.text_processor = TextProcessor()
        self.hybrid_search = HybridSearchEngine()
        self.faiss_manager = FAISSIndexManager(Config.FAISS_INDEX_PATH, Config.FAISS_METADATA_PATH)

        # Try to load existing index
        self.faiss_manager.load_index()

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence-transformers (local, fast, free)"""
        if not texts:
            return np.array([])

        # Always use local sentence-transformers for embeddings (fast & free)
        return self.embedder.encode(texts).astype('float32')

    def update_faiss_index(self, papers: List[Dict]) -> int:
        """
        Incrementally update FAISS index with only new papers
        Returns number of papers added
        """
        # Filter out already indexed papers
        new_papers = self.faiss_manager.get_new_papers(papers)

        if not new_papers:
            print("[INDEX] All papers already indexed - reusing existing index")
            return 0

        print(f"[INDEX] Indexing {len(new_papers)} new papers...")

        # Get abstracts for embedding
        abstracts = [p.get('abstract', '') or p.get('title', '') for p in new_papers]

        # Generate embeddings
        new_embeddings = self.generate_embeddings(abstracts)

        if new_embeddings.size == 0:
            return 0

        # Normalize for cosine similarity
        faiss.normalize_L2(new_embeddings)

        # Create or update index
        if self.faiss_manager.index is None:
            # Create new index
            d = new_embeddings.shape[1]
            self.faiss_manager.index = faiss.IndexFlatIP(d)

        # Add to index
        self.faiss_manager.index.add(new_embeddings)

        # Save papers to database and mark as indexed
        for paper in new_papers:
            self.paper_manager.save_paper(paper)
            self.faiss_manager.mark_as_indexed(paper.get('title', ''))

        # Persist index
        self.faiss_manager.save_index()

        return len(new_papers)

    def hybrid_retrieve(self, query: str, papers: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """
        Hybrid retrieval combining semantic and keyword search
        """
        # ANNOTATED: was `top_k: int = None`, which is invalid — None is not an int
        top_k = top_k or Config.TOP_K_RETRIEVAL

        if not papers:
            return []

        # Generate query embedding
        query_emb = self.generate_embeddings([query])
        if query_emb.size == 0:
            return papers[:top_k]

        faiss.normalize_L2(query_emb)

        # Generate paper embeddings
        abstracts = [p.get('abstract', '') or p.get('title', '') for p in papers]
        paper_embs = self.generate_embeddings(abstracts)

        if paper_embs.size == 0:
            return papers[:top_k]

        faiss.normalize_L2(paper_embs)

        # Calculate semantic similarity
        semantic_scores = np.dot(paper_embs, query_emb.T).flatten()

        # Calculate hybrid scores
        # ANNOTATED: mypy cannot infer element type from an empty list literal
        scored_papers: List[Dict] = []
        for i, paper in enumerate(papers):
            # Semantic score
            sem_score = float(semantic_scores[i]) if i < len(semantic_scores) else 0.0

            # Keyword score
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            kw_score = self.hybrid_search.calculate_keyword_score(query, text)

            # Hybrid score
            hybrid_score = self.hybrid_search.calculate_hybrid_score(sem_score, kw_score)

            # Add citation boost (normalized)
            citations = paper.get('citation_count', 0) or 0
            citation_boost = min(citations / 1000, 0.1)  # Max 10% boost

            paper_copy = paper.copy()
            paper_copy['semantic_score'] = sem_score
            paper_copy['keyword_score'] = kw_score
            paper_copy['similarity_score'] = hybrid_score + citation_boost

            scored_papers.append(paper_copy)

        # Sort by hybrid score
        sorted_papers = sorted(scored_papers, key=lambda x: x['similarity_score'], reverse=True)

        # Add rank
        for i, paper in enumerate(sorted_papers):
            paper['rank'] = i + 1

        return sorted_papers[:top_k]

    # ==================== AI RESPONSE METHODS (GPT-4o-mini) ====================

    def _build_context(self, papers: List[Dict], max_papers: int = 8) -> str:
        """Build context from papers for AI prompt"""
        # ANNOTATED: mypy cannot infer element type from an empty list literal
        context_parts: List[str] = []
        for i, paper in enumerate(papers[:max_papers]):
            authors = ', '.join(paper.get('authors', [])[:3])
            if len(paper.get('authors', [])) > 3:
                authors += ' et al.'

            abstract = paper.get('abstract', '')[:500]
            year = paper.get('year', 'N/A')
            citations = paper.get('citation_count', 'N/A')

            context_parts.append(
                f"Paper {i+1}: {paper['title']}\n"
                f"Authors: {authors}\n"
                f"Year: {year} | Citations: {citations}\n"
                f"Abstract: {abstract}..."
            )

        return "\n\n".join(context_parts)

    def generate_summary_stream(self, query: str, papers: List[Dict]) -> Generator[str, None, None]:
        """Generate comprehensive literature summary with streaming"""
        if not papers:
            yield "No relevant papers found for summarization."
            return

        context = self._build_context(papers)

        prompt = f"""You are an expert research communicator and mentor. Based on the following research papers, produce a clear, engaging, and human-friendly research analysis about: {query}

PAPERS:
{context}

WRITING STYLE & TONE GUIDELINES:
- Write in simple, clear, engaging, and accessible language so that anyone (students, non-specialists, and researchers) can understand it easily.
- Avoid overly dense or impenetrable academic jargon. If you introduce a technical term, explain what it means in plain everyday words.
- Use smooth transitions and crisp, digestible sentences.
- Make the insights practical, interesting, and relatable while preserving scientific accuracy and depth.
- Cite papers naturally using (Author, Year) format.

IMPORTANT FORMATTING RULES:
Do NOT use Markdown symbols (no #, *, -, or bullet points).
Do NOT use emojis.
Use plain text only.
Do NOT use vertical lines, double pipes (||), or hyphens inside words (write 'Cross lingual', 'real world', 'state of the art', 'machine learning' without hyphens).
Use clear section headings in BOLD UPPERCASE (e.g., AI ANALYSIS, LITERATURE SUMMARY).
Each section must be written in continuous, well-structured paragraph form.
Keep formatting clean and easy to read.

Required Structure:

AI ANALYSIS
Provide a clear, high-level big picture summary of the topic and key insights from the analyzed papers. Explain what is happening in this field in simple, engaging terms and highlight the most important discoveries.

LITERATURE SUMMARY
Explain what past and current studies have discovered in simple, straightforward language. Cover traditional approaches alongside modern AI and data-driven methods. Mention key contributions from prior studies and cite relevant papers.

RESEARCH GAPS
Explain what current research is still missing or struggling with. Discuss real-world challenges like scalability, data quality, bias, lack of explainability, and why lab models often struggle when deployed in everyday life.

COMPARISON OF STUDIES
Compare the different research approaches in a clear, balanced way. Contrast older classical methods with modern AI techniques, highlighting what each approach does well and where each falls short.

METHODOLOGIES
Explain the key methods and techniques used across the studies in simple, intuitive terms. Demystify how researchers collected data, trained models, and evaluated their results.

APPLICATIONS AND IMPLICATIONS
Describe the practical, real-world impact of these findings. Explain how this research affects everyday society, industry, technology, and public policy.

CHALLENGES AND LIMITATIONS
Discuss the practical roadblocks and hurdles researchers face, such as computing costs, privacy barriers, lack of data, and technical limitations, in plain English.

FUTURE DIRECTIONS
Outline exciting future opportunities and where the field is heading next. Highlight what researchers and developers should focus on to solve remaining problems."""

        yield from self._stream_ai_response(prompt)

    def identify_research_gaps_stream(self, query: str, papers: List[Dict]) -> Generator[str, None, None]:
        """Identify research gaps with streaming"""
        if not papers:
            yield "No papers available for gap analysis."
            return

        # Filter recent papers
        # ANNOTATED (bug guard): int(p['year']) can raise ValueError on non-numeric
        # year values (e.g. "N/A", ""), so this is wrapped safely below.
        def _is_recent(p: Dict) -> bool:
            try:
                return bool(p.get('year')) and int(p['year']) >= 2020
            except (TypeError, ValueError):
                return False

        recent_papers = [p for p in papers if _is_recent(p)]
        papers_to_analyze = recent_papers[:8] if recent_papers else papers[:8]

        context = self._build_context(papers_to_analyze)

        prompt = f"""You are an expert research communicator and mentor. Analyze these recent papers about "{query}" and produce a clear, human-friendly research gap analysis that anyone can easily understand.

PAPERS:
{context}

WRITING STYLE & TONE GUIDELINES:
- Write in clear, engaging, and accessible language that is easy for anyone to understand.
- Avoid dense academic jargon; explain technical concepts simply and clearly.
- Keep sentences crisp, lively, and easy to follow.
- Cite papers naturally using (Author, Year) format.

IMPORTANT FORMATTING RULES:
Do NOT use Markdown symbols (no #, *, -, or bullet points).
Do NOT use emojis.
Use plain text only.
Do NOT use vertical lines, double pipes (||), or hyphens inside words (write 'Cross lingual', 'real world', 'state of the art', 'machine learning' without hyphens).
Use clear section headings in BOLD UPPERCASE (e.g., AI ANALYSIS, RESEARCH GAPS).
Each section must be written in continuous, well-structured paragraph form.
Keep formatting clean and easy to read.

Required Structure:

AI ANALYSIS
Provide a clear overview of the current state of research and why identifying research gaps in this field matters in simple, relatable terms.

LITERATURE SUMMARY
Summarize what existing studies have accomplished so far in simple terms, citing key papers.

RESEARCH GAPS
Clearly explain the major unanswered questions and missing pieces in current research, such as untested scenarios, missing datasets, and real-world shortcomings.

COMPARISON OF STUDIES
Compare the identified gaps by showing which ones are critical vs minor, and which research problems need immediate attention versus long-term solutions.

METHODOLOGIES
Explain the methods used in these studies in simple terms and where new, innovative techniques are needed.

APPLICATIONS AND IMPLICATIONS
Describe the real-world benefits and opportunities that will open up once these research gaps are solved.

CHALLENGES AND LIMITATIONS
Discuss the practical difficulties and real-world barriers that make these gaps tricky to address.

FUTURE DIRECTIONS
Suggest practical, actionable next steps and project ideas for future research."""

        yield from self._stream_ai_response(prompt)

    def compare_studies_stream(self, papers: List[Dict]) -> Generator[str, None, None]:
        """Compare studies with streaming"""
        if len(papers) < 2:
            yield "Need at least 2 papers for comparison."
            return

        # Build comparison table
        # ANNOTATED: mypy cannot infer element type from an empty list literal
        comparison_entries: List[str] = []
        for i, paper in enumerate(papers[:6]):
            methodology = self.text_processor.extract_methodology(paper.get('abstract', ''))
            comparison_entries.append(
                f"Study {i+1}: {paper['title'][:100]}\n"
                f"  Year: {paper.get('year', 'N/A')}\n"
                f"  Methodology: {methodology}\n"
                f"  Citations: {paper.get('citation_count', 'N/A')}"
            )

        prompt = f"""You are an expert research communicator. Compare these research studies and produce a clear, human-friendly comparative analysis that is easy for anyone to understand:

{chr(10).join(comparison_entries)}

WRITING STYLE & TONE GUIDELINES:
- Write in simple, accessible, and engaging language.
- Explain technical terms and methodologies in plain English.
- Highlight the real-world strengths and weaknesses of each study.
- Cite studies using (Author, Year) format.

IMPORTANT FORMATTING RULES:
Do NOT use Markdown symbols (no #, *, -, or bullet points).
Do NOT use emojis.
Use plain text only.
Do NOT use vertical lines, double pipes (||), or hyphens inside words (write 'Cross lingual', 'real world', 'state of the art', 'machine learning' without hyphens).
Use clear section headings in BOLD UPPERCASE (e.g., AI ANALYSIS, COMPARISON OF STUDIES).
Each section must be written in continuous, well-structured paragraph form.
Keep formatting clean and easy to read.

Required Structure:

AI ANALYSIS
Provide a clear, high-level overview of the studies being compared and the main takeaways in accessible language.

LITERATURE SUMMARY
Summarize the common themes, timeline, and progression of knowledge across these studies in easy-to-understand terms.

RESEARCH GAPS
Identify limitations and unanswered questions that become visible when comparing these studies side-by-side.

COMPARISON OF STUDIES
Compare the strengths, trade-offs, and differences between the studies in a clear and conversational manner.

METHODOLOGIES
Explain the different techniques and approaches used by each study in simple, intuitive terms.

APPLICATIONS AND IMPLICATIONS
Explain how the collective findings can be used in the real world and what they mean for the industry.

CHALLENGES AND LIMITATIONS
Highlight common hurdles, shared constraints, and potential flaws across the studies.

FUTURE DIRECTIONS
Provide a forward-looking conclusion on which approaches are most promising and where researchers should focus next."""

        yield from self._stream_ai_response(prompt)

    def suggest_methodologies_stream(self, query: str, papers: List[Dict]) -> Generator[str, None, None]:
        """Suggest methodologies with streaming"""
        # Extract existing methods
        # ANNOTATED: mypy cannot infer the element type from an empty `set()`
        existing_methods: Set[str] = set()
        for paper in papers[:10]:
            methods = self.text_processor.extract_methodology(paper.get('abstract', ''))
            if methods and methods != 'Not specified':
                existing_methods.update(m.strip() for m in methods.split(','))

        methods_str = ', '.join(existing_methods) if existing_methods else 'Various approaches'

        prompt = f"""You are an expert research communicator and advisor. For research on: "{query}"

Existing methodologies found in literature: {methods_str}

Suggest innovative, practical research methodologies explained in clear, human-friendly terms that anyone can easily understand.

WRITING STYLE & TONE GUIDELINES:
- Explain all techniques in plain, engaging English without overwhelming jargon.
- Use intuitive examples and clear explanations of how each method works in practice.
- Keep the writing energetic, helpful, and accessible to students and professionals alike.

IMPORTANT FORMATTING RULES:
Do NOT use Markdown symbols (no #, *, -, or bullet points).
Do NOT use emojis.
Use plain text only.
Do NOT use vertical lines, double pipes (||), or hyphens inside words (write 'Cross lingual', 'real world', 'state of the art', 'machine learning' without hyphens).
Use clear section headings in BOLD UPPERCASE (e.g., AI ANALYSIS, METHODOLOGIES).
Each section must be written in continuous, well-structured paragraph form.
Keep formatting clean and easy to read.

Required Structure:

AI ANALYSIS
Provide a simple, engaging overview of how researchers currently study this topic and why choosing the right method matters.

LITERATURE SUMMARY
Summarize the existing methods currently used in the field in plain English.

RESEARCH GAPS
Explain what current research tools and methods are missing or failing to capture.

COMPARISON OF STUDIES
Compare traditional methods with modern AI techniques, explaining the pros, cons, and best use cases of each.

METHODOLOGIES
Explain recommended methodologies clearly in simple terms, covering classical techniques, modern AI tools, and smart hybrid approaches.

APPLICATIONS AND IMPLICATIONS
Describe how applying these methods creates real-world value and practical impact.

CHALLENGES AND LIMITATIONS
Explain the practical challenges (like computational cost, data needs, and setup requirements) in simple terms.

FUTURE DIRECTIONS
Highlight the most exciting upcoming methodological trends and innovations to watch."""

        yield from self._stream_ai_response(prompt)

    def _stream_ai_response(self, prompt: str) -> Generator[str, None, None]:
        """
        Stream AI response via the Multi-Provider LLM Router.

        Provider cascade: Gemini → Groq → Ollama → Safe Fallback
        The router handles ALL error handling, rate limiting, and fallback logic.
        This method will NEVER raise an exception.
        """
        try:
            yield from self.router.generate_stream(
                prompt=prompt,
                temperature=Config.AI_TEMPERATURE,
                max_tokens=Config.AI_MAX_TOKENS
            )

            # Log which provider was used
            provider = self.router.last_provider_used or 'unknown'
            print(f"[OK] AI response streamed via: {provider}")

        except Exception as e:
            # This should never happen (router handles all errors), but just in case
            print(f"[ERROR] Unexpected router error: {e}")
            yield (
                "\n\nNote: An unexpected error occurred. "
                "Please try again or check your API configuration."
            )

    # ==================== NON-STREAMING METHODS (for backward compatibility) ====================

    def generate_summary(self, query: str, papers: List[Dict]) -> str:
        """Non-streaming summary generation"""
        return ''.join(self.generate_summary_stream(query, papers))

    def identify_research_gaps(self, query: str, papers: List[Dict]) -> str:
        """Non-streaming gap analysis"""
        return ''.join(self.identify_research_gaps_stream(query, papers))

    def compare_studies(self, papers: List[Dict]) -> str:
        """Non-streaming study comparison"""
        return ''.join(self.compare_studies_stream(papers))

    def suggest_methodologies(self, query: str, papers: List[Dict]) -> str:
        """Non-streaming methodology suggestions"""
        return ''.join(self.suggest_methodologies_stream(query, papers))

    # ==================== MAIN PROCESSING ====================

    def process_query(self, query: str, feature: str, papers: List[Dict]) -> Dict[str, Any]:
        """Main processing function with hybrid search"""
        if not papers:
            return {"error": "No papers found for the given query.", "success": False}

        print(f"\n[RAG] Processing query: '{query}' | Feature: {feature}")

        # Update FAISS index with new papers only
        new_count = self.update_faiss_index(papers)
        print(f"   Added {new_count} new papers to index")

        # Perform hybrid retrieval
        retrieved_papers = self.hybrid_retrieve(query, papers)
        print(f"   Retrieved {len(retrieved_papers)} relevant papers")

        # Generate response based on feature
        try:
            if feature == 'summarize':
                result = self.generate_summary(query, retrieved_papers)
            elif feature == 'gaps':
                result = self.identify_research_gaps(query, retrieved_papers)
            elif feature == 'compare':
                result = self.compare_studies(retrieved_papers)
            elif feature == 'methods':
                result = self.suggest_methodologies(query, retrieved_papers)
            else:
                result = "Unknown feature selected."

            # Clean and sanitize report text for human-friendly presentation
            clean_result = clean_report_text(result)

            return {
                "success": True,
                "result": clean_result,
                "papers_analyzed": len(retrieved_papers),
                "top_papers": retrieved_papers[:5]
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Processing error: {str(e)}"
            }

    def process_query_stream(self, query: str, feature: str, papers: List[Dict]) -> Generator[str, None, None]:
        """Streaming version of process_query for SSE"""
        if not papers:
            yield "data: No papers found for the given query.\n\n"
            return

        # Update FAISS index
        self.update_faiss_index(papers)

        # Perform hybrid retrieval
        retrieved_papers = self.hybrid_retrieve(query, papers)

        # Stream based on feature
        if feature == 'summarize':
            yield from self.generate_summary_stream(query, retrieved_papers)
        elif feature == 'gaps':
            yield from self.identify_research_gaps_stream(query, retrieved_papers)
        elif feature == 'compare':
            yield from self.compare_studies_stream(retrieved_papers)
        elif feature == 'methods':
            yield from self.suggest_methodologies_stream(query, retrieved_papers)
        else:
            yield "Unknown feature selected."


# Global instance
rag_pipeline = RAGPipeline()