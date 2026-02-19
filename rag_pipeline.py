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
from typing import List, Dict, Any, Generator, Optional
from llm_router import llm_router


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
        self.index = None
        self.indexed_titles = set()
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
    
    def load_index(self) -> Optional[faiss.Index]:
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
        print(f"🔄 RAG Pipeline using Multi-Provider LLM Router")
        
        # Show provider status
        status = self.router.get_status()
        for pname, pinfo in status['providers'].items():
            icon = '✅' if pinfo['available'] else '⚠️'
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
            print("📊 All papers already indexed - reusing existing index")
            return 0
        
        print(f"📊 Indexing {len(new_papers)} new papers...")
        
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
    
    def hybrid_retrieve(self, query: str, papers: List[Dict], top_k: int = None) -> List[Dict]:
        """
        Hybrid retrieval combining semantic and keyword search
        """
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
        scored_papers = []
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
        context_parts = []
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
        
        prompt = f"""You are an expert academic research assistant. Based on the following research papers, write a comprehensive, formal academic literature review about: {query}

PAPERS:
{context}

IMPORTANT FORMATTING INSTRUCTIONS:
- Write in formal academic English using continuous, well-structured paragraphs only.
- Do NOT use any markdown formatting (no ##, **, *, -, etc.).
- Do NOT use emojis, hashtags, bullet points, numbered lists, or section titles.
- Do NOT use labels like "Paper 1" or "[1]". Instead, cite each paper using the (Author, Year) format based on the authors and year provided above. For example: (Smith et al., 2021).
- The writing should read like a literature review section of a final year project or research paper.
- Ensure the tone is natural, human-like, and suitable for academic publication.
- Cover the following aspects seamlessly within the paragraphs: main research themes across the literature, key findings and significant contributions, common methodologies and research approaches, temporal trends and emerging directions, and an integrated synthesis of the current state of research.
- Maintain coherence, clarity, and smooth transitions between ideas.
- The review should be plagiarism-free and demonstrate critical analysis rather than mere description."""

        yield from self._stream_ai_response(prompt)
    
    def identify_research_gaps_stream(self, query: str, papers: List[Dict]) -> Generator[str, None, None]:
        """Identify research gaps with streaming"""
        if not papers:
            yield "No papers available for gap analysis."
            return
        
        # Filter recent papers
        recent_papers = [p for p in papers if p.get('year') and int(p['year']) >= 2020]
        papers_to_analyze = recent_papers[:8] if recent_papers else papers[:8]
        
        context = self._build_context(papers_to_analyze)
        
        prompt = f"""You are an expert research mentor. Analyze these recent papers about "{query}" and write a comprehensive, formal academic research gap analysis.

PAPERS:
{context}

IMPORTANT FORMATTING INSTRUCTIONS:
- Write in formal academic English using continuous, well-structured paragraphs only.
- Do NOT use any markdown formatting (no ##, **, *, -, etc.).
- Do NOT use emojis, hashtags, bullet points, numbered lists, or section titles.
- Do NOT use labels like "Paper 1" or "[1]". Instead, cite each paper using the (Author, Year) format based on the authors and year provided above. For example: (Smith et al., 2021).
- The writing should read like a research gap analysis section of a final year project or research paper.
- Ensure the tone is natural, human-like, and suitable for academic publication.
- Cover the following aspects seamlessly within the paragraphs: unexplored research questions that have not been adequately addressed, methodological gaps and underutilised techniques, data and contextual gaps including missing populations or datasets, emerging opportunities and promising new research directions, and a recommended research agenda with specific proposed projects.
- Maintain coherence, clarity, and smooth transitions between ideas.
- Be specific, actionable, and demonstrate critical analysis."""

        yield from self._stream_ai_response(prompt)
    
    def compare_studies_stream(self, papers: List[Dict]) -> Generator[str, None, None]:
        """Compare studies with streaming"""
        if len(papers) < 2:
            yield "Need at least 2 papers for comparison."
            return
        
        # Build comparison table
        comparison_entries = []
        for i, paper in enumerate(papers[:6]):
            methodology = self.text_processor.extract_methodology(paper.get('abstract', ''))
            comparison_entries.append(
                f"Study {i+1}: {paper['title'][:100]}\n"
                f"  - Year: {paper.get('year', 'N/A')}\n"
                f"  - Methodology: {methodology}\n"
                f"  - Citations: {paper.get('citation_count', 'N/A')}"
            )
        
        prompt = f"""You are an expert in systematic literature reviews. Compare these research studies and write a formal academic comparative analysis:

{chr(10).join(comparison_entries)}

IMPORTANT FORMATTING INSTRUCTIONS:
- Write in formal academic English using continuous, well-structured paragraphs only.
- Do NOT use any markdown formatting (no ##, **, *, -, etc.).
- Do NOT use emojis, hashtags, bullet points, numbered lists, or section titles.
- Do NOT use labels like "Study 1" or "[1]". Instead, cite each study using the (Author, Year) format based on the authors and year provided above. For example: (Smith et al., 2021).
- The writing should read like a comparative analysis section of a final year project or research paper.
- Ensure the tone is natural, human-like, and suitable for academic publication.
- Cover the following aspects seamlessly within the paragraphs: comparison of research designs, data sources, and analytical approaches across the studies, areas of agreement and divergence in findings, unique contributions and strengths of each study, common and study-specific limitations, and an overall synthesis drawing conclusions across all studies.
- Maintain coherence, clarity, and smooth transitions between ideas.
- Be balanced, evidence-based, and demonstrate critical analysis."""

        yield from self._stream_ai_response(prompt)
    
    def suggest_methodologies_stream(self, query: str, papers: List[Dict]) -> Generator[str, None, None]:
        """Suggest methodologies with streaming"""
        # Extract existing methods
        existing_methods = set()
        for paper in papers[:10]:
            methods = self.text_processor.extract_methodology(paper.get('abstract', ''))
            if methods and methods != 'Not specified':
                existing_methods.update(m.strip() for m in methods.split(','))
        
        methods_str = ', '.join(existing_methods) if existing_methods else 'Various approaches'
        
        prompt = f"""You are a research methodology expert. For research on: "{query}"

Existing methodologies found in literature: {methods_str}

Suggest innovative and appropriate research methodologies in a formal academic writing style.

IMPORTANT FORMATTING INSTRUCTIONS:
- Write in formal academic English using continuous, well-structured paragraphs only.
- Do NOT use any markdown formatting (no ##, **, *, -, etc.).
- Do NOT use emojis, hashtags, bullet points, numbered lists, or section titles.
- The writing should read like a methodology discussion section of a final year project or research paper.
- Ensure the tone is natural, human-like, and suitable for academic publication.
- Cover the following aspects seamlessly within the paragraphs: well-established traditional approaches with modern adaptations, cutting-edge and emerging methodologies, integrated mixed methods designs combining multiple approaches, computational and AI-enhanced research methods, and implementation recommendations including applicability, key requirements, and potential benefits and limitations for each suggested methodology.
- Maintain coherence, clarity, and smooth transitions between ideas.
- Be specific, practical, and demonstrate scholarly depth in your suggestions."""

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
            print(f"✅ AI response streamed via: {provider}")
            
        except Exception as e:
            # This should never happen (router handles all errors), but just in case
            print(f"❌ Unexpected router error: {e}")
            yield (
                "\n\n---\n*⚠️ An unexpected error occurred. "
                "Please try again or check your API configuration.*"
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
        
        print(f"\n🔄 Processing query: '{query}' | Feature: {feature}")
        
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
            
            return {
                "success": True,
                "result": result,
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