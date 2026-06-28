# api_integrations.py
"""
Enhanced API Integrations with Async Parallel Fetching
Uses asyncio + httpx for true concurrent I/O across all APIs
"""

import asyncio
import httpx
import requests
import json
import os
import time
import arxiv
from pymed import PubMed
from config import Config
from typing import List, Dict, Any
from urllib.parse import quote
import threading


# Per-API hard timeouts (seconds)
# Fast APIs get more time, slow/unreliable APIs get cut off early
API_TIMEOUTS = {
    'arxiv': 10,
    'semantic_scholar': 10,
    'openalex': 10,
    'pubmed': 8,
    'crossref': 8,
    'core': 6,
    'doaj': 6,
    'europe_pmc': 6,
}


class SearchProgressTracker:
    """Track search progress for real-time UI updates"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._status = {}
        self._results = {}
        self._start_time = None
    
    def start(self):
        """Start tracking search progress"""
        with self._lock:
            self._status = {}
            self._results = {}
            self._start_time = time.time()
    
    def update_status(self, api_name: str, status: str, count: int = 0):
        """Update status for an API"""
        with self._lock:
            self._status[api_name] = {
                'status': status,
                'count': count,
                'timestamp': time.time()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current search status"""
        with self._lock:
            elapsed = time.time() - self._start_time if self._start_time else 0
            return {
                'apis': self._status.copy(),
                'elapsed_time': round(elapsed, 2)
            }
    
    def set_results(self, api_name: str, papers: List[Dict]):
        """Store results from an API"""
        with self._lock:
            self._results[api_name] = papers


# Global progress tracker for SSE
search_progress = SearchProgressTracker()


class APIManager:
    """
    Enhanced API Manager with Async Parallel Fetching
    Uses asyncio + httpx for true concurrent I/O with per-API timeouts
    """
    
    def __init__(self):
        # Keep requests session for sync library-based APIs (arxiv, pymed)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AcademicResearchAssistant/2.0',
            'Accept': 'application/json'
        })
        
        # Shared headers for async httpx client
        self._async_headers = {
            'User-Agent': 'AcademicResearchAssistant/2.0',
            'Accept': 'application/json'
        }
        
        # API endpoints
        self.core_api_url = "https://api.core.ac.uk/v3/search/works"
        self.crossref_api_url = "https://api.crossref.org/works"
        self.openalex_api_url = "https://api.openalex.org/works"
        self.doaj_api_url = "https://doaj.org/api/search/articles"
        self.europe_pmc_api_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        
        # API display names for progress tracking
        self.api_display_names = {
            'arxiv': 'arXiv',
            'pubmed': 'PubMed',
            'semantic_scholar': 'Semantic Scholar',
            'core': 'CORE',
            'crossref': 'CrossRef',
            'openalex': 'OpenAlex',
            'doaj': 'DOAJ',
            'europe_pmc': 'Europe PMC'
        }
    
    # ==================== CACHING METHODS ====================
    
    def cache_response(self, query: str, api_name: str, response: Any) -> None:
        """Enhanced caching with metadata — normalizes query for consistent keys"""
        query = query.lower().strip()
        try:
            cache_file = os.path.join(Config.CACHE_DIR, f"{api_name}_{quote(query, safe='')}.json")
            cache_data = {
                'timestamp': time.time(),
                'query': query,
                'data': response
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Cache write error for {api_name}: {e}")
    
    def load_cached(self, query: str, api_name: str) -> Any:
        """Load cached response if valid — normalizes query for consistent keys"""
        query = query.lower().strip()
        try:
            cache_file = os.path.join(Config.CACHE_DIR, f"{api_name}_{quote(query, safe='')}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                # Check if cache is still valid
                if time.time() - cache_data['timestamp'] < Config.CACHE_TIMEOUT:
                    return cache_data['data']
        except Exception as e:
            print(f"Cache read error for {api_name}: {e}")
        return None
    
    # ==================== SYNC API METHODS (library-based) ====================
    # These use third-party libraries that are inherently synchronous.
    # They are wrapped in asyncio.to_thread() when called from the async parallel method.
    
    def fetch_arxiv(self, query: str, max_results: int = None) -> List[Dict]:
        """Enhanced arXiv API with proper parsing (sync — uses arxiv library)"""
        max_results = max_results or Config.MAX_PAPERS_PER_API
        
        cached = self.load_cached(query, 'arxiv')
        if cached:
            search_progress.update_status('arxiv', 'cached', len(cached))
            return cached
        
        search_progress.update_status('arxiv', 'searching')
        
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in client.results(search):
                paper = {
                    'source': 'arxiv',
                    'title': result.title,
                    'abstract': result.summary or '',
                    'authors': [author.name for author in result.authors],
                    'year': result.published.year if result.published else None,
                    'doi': result.doi,
                    'pdf_url': result.pdf_url,
                    'url': result.entry_id,
                    'published': result.published.isoformat() if result.published else None,
                    'categories': result.categories,
                    'citation_count': 0
                }
                papers.append(paper)
            
            self.cache_response(query, 'arxiv', papers)
            search_progress.update_status('arxiv', 'completed', len(papers))
            return papers
            
        except Exception as e:
            print(f"arXiv API error: {e}")
            search_progress.update_status('arxiv', 'failed', 0)
            return []
    
    def fetch_pubmed(self, query: str, max_results: int = None) -> List[Dict]:
        """Enhanced PubMed API (sync — uses pymed library)"""
        max_results = max_results or Config.MAX_PAPERS_PER_API
        
        cached = self.load_cached(query, 'pubmed')
        if cached:
            search_progress.update_status('pubmed', 'cached', len(cached))
            return cached
        
        search_progress.update_status('pubmed', 'searching')
        
        try:
            pubmed = PubMed(tool="AcademicResearchAssistant", email="research@assistant.com")
            results = pubmed.query(query, max_results=max_results)
            
            papers = []
            for article in results:
                try:
                    # Handle author extraction
                    authors = []
                    if hasattr(article, 'authors') and article.authors:
                        for author in article.authors:
                            if isinstance(author, dict):
                                name = f"{author.get('firstname', '')} {author.get('lastname', '')}".strip()
                            else:
                                name = str(author)
                            if name:
                                authors.append(name)
                    
                    paper = {
                        'source': 'pubmed',
                        'title': str(article.title) if article.title else '',
                        'abstract': str(article.abstract) if article.abstract else '',
                        'authors': authors,
                        'year': article.publication_date.year if hasattr(article, 'publication_date') and article.publication_date else None,
                        'doi': str(article.doi) if hasattr(article, 'doi') and article.doi else None,
                        'journal': str(article.journal) if hasattr(article, 'journal') and article.journal else None,
                        'pubmed_id': str(article.pubmed_id) if hasattr(article, 'pubmed_id') else None,
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{article.pubmed_id}/" if hasattr(article, 'pubmed_id') and article.pubmed_id else None,
                        'citation_count': 0
                    }
                    if paper['title']:
                        papers.append(paper)
                except Exception as e:
                    continue
            
            self.cache_response(query, 'pubmed', papers)
            search_progress.update_status('pubmed', 'completed', len(papers))
            return papers
            
        except Exception as e:
            print(f"PubMed API error: {e}")
            search_progress.update_status('pubmed', 'failed', 0)
            return []
    
    # ==================== ASYNC API METHODS (httpx-based) ====================
    # These use httpx.AsyncClient for true async I/O.
    # The client is created once in fetch_papers_parallel and passed in.
    
    async def fetch_semantic_scholar(self, query: str, client: httpx.AsyncClient, max_results: int = None) -> List[Dict]:
        """Enhanced Semantic Scholar API (async)"""
        max_results = max_results or Config.MAX_PAPERS_PER_API
        
        cached = self.load_cached(query, 'semantic_scholar')
        if cached:
            search_progress.update_status('semantic_scholar', 'cached', len(cached))
            return cached
        
        search_progress.update_status('semantic_scholar', 'searching')
        
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': query,
                'limit': max_results,
                'fields': 'title,abstract,authors,year,doi,venue,url,citationCount'
            }
            
            headers = {}
            if Config.SEMANTIC_SCHOLAR_API_KEY:
                headers['x-api-key'] = Config.SEMANTIC_SCHOLAR_API_KEY
            
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            for item in data.get('data', []):
                paper = {
                    'source': 'semantic_scholar',
                    'title': item.get('title', ''),
                    'abstract': item.get('abstract', '') or '',
                    'authors': [author.get('name', '') for author in item.get('authors', []) if author.get('name')],
                    'year': item.get('year'),
                    'doi': item.get('doi'),
                    'venue': item.get('venue'),
                    'url': item.get('url'),
                    'citation_count': item.get('citationCount', 0) or 0
                }
                if paper['title']:
                    papers.append(paper)
            
            self.cache_response(query, 'semantic_scholar', papers)
            search_progress.update_status('semantic_scholar', 'completed', len(papers))
            return papers
            
        except Exception as e:
            print(f"Semantic Scholar API error: {e}")
            search_progress.update_status('semantic_scholar', 'failed', 0)
            return []
    
    async def fetch_core(self, query: str, client: httpx.AsyncClient, max_results: int = None) -> List[Dict]:
        """CORE API - One of the largest collections of open access research papers (async)"""
        max_results = max_results or Config.MAX_PAPERS_PER_API
        
        cached = self.load_cached(query, 'core')
        if cached:
            search_progress.update_status('core', 'cached', len(cached))
            return cached
        
        search_progress.update_status('core', 'searching')
        
        try:
            params = {
                'q': query,
                'limit': max_results,
            }
            
            headers = {}
            if hasattr(Config, 'CORE_API_KEY') and Config.CORE_API_KEY:
                headers['Authorization'] = f'Bearer {Config.CORE_API_KEY}'
            
            response = await client.get(self.core_api_url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            for item in data.get('results', []):
                urls = item.get('sourceFulltextUrls', [])
                paper = {
                    'source': 'core',
                    'title': item.get('title', ''),
                    'abstract': item.get('abstract', '') or '',
                    'authors': [author.get('name', '') for author in item.get('authors', []) if author.get('name')],
                    'year': item.get('yearPublished'),
                    'doi': item.get('doi'),
                    'url': item.get('downloadUrl') or (urls[0] if urls else None),
                    'publisher': item.get('publisher'),
                    'journal': item.get('journals', [{}])[0].get('title') if item.get('journals') else None,
                    'citation_count': 0
                }
                if paper['title']:
                    papers.append(paper)
            
            self.cache_response(query, 'core', papers)
            search_progress.update_status('core', 'completed', len(papers))
            return papers
            
        except Exception as e:
            print(f"CORE API error: {e}")
            search_progress.update_status('core', 'failed', 0)
            return []
    
    async def fetch_crossref(self, query: str, client: httpx.AsyncClient, max_results: int = None) -> List[Dict]:
        """CrossRef API - Primary source for scientific metadata (async)"""
        max_results = max_results or Config.MAX_PAPERS_PER_API
        
        cached = self.load_cached(query, 'crossref')
        if cached:
            search_progress.update_status('crossref', 'cached', len(cached))
            return cached
        
        search_progress.update_status('crossref', 'searching')
        
        try:
            params = {
                'query': query,
                'rows': max_results,
                'select': 'title,author,abstract,DOI,published-print,published-online,container-title,URL,subject'
            }
            
            response = await client.get(self.crossref_api_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            for item in data.get('message', {}).get('items', []):
                # Extract year from published date
                year = None
                if item.get('published-print'):
                    date_parts = item['published-print'].get('date-parts', [[None]])
                    year = date_parts[0][0] if date_parts and date_parts[0] else None
                elif item.get('published-online'):
                    date_parts = item['published-online'].get('date-parts', [[None]])
                    year = date_parts[0][0] if date_parts and date_parts[0] else None
                
                paper = {
                    'source': 'crossref',
                    'title': item.get('title', [''])[0] if item.get('title') else '',
                    'abstract': item.get('abstract', '') or '',
                    'authors': [f"{author.get('given', '')} {author.get('family', '')}".strip() 
                               for author in item.get('author', [])],
                    'year': year,
                    'doi': item.get('DOI'),
                    'url': item.get('URL'),
                    'journal': item.get('container-title', [''])[0] if item.get('container-title') else None,
                    'subjects': item.get('subject', []),
                    'citation_count': item.get('is-referenced-by-count', 0) or 0
                }
                if paper['title']:
                    papers.append(paper)
            
            self.cache_response(query, 'crossref', papers)
            search_progress.update_status('crossref', 'completed', len(papers))
            return papers
            
        except Exception as e:
            print(f"CrossRef API error: {e}")
            search_progress.update_status('crossref', 'failed', 0)
            return []
    
    async def fetch_openalex(self, query: str, client: httpx.AsyncClient, max_results: int = None) -> List[Dict]:
        """OpenAlex API - Free, open catalog of the global research system (async)"""
        max_results = max_results or Config.MAX_PAPERS_PER_API
        
        cached = self.load_cached(query, 'openalex')
        if cached:
            search_progress.update_status('openalex', 'cached', len(cached))
            return cached
        
        search_progress.update_status('openalex', 'searching')
        
        try:
            params = {
                'search': query,
                'per-page': max_results,
                'mailto': 'research-assistant@academic.edu'
            }
            
            response = await client.get(self.openalex_api_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            for item in data.get('results', []):
                primary_location = item.get('primary_location') or {}
                source = primary_location.get('source') or {}
                
                paper = {
                    'source': 'openalex',
                    'title': item.get('title', ''),
                    'abstract': self._get_openalex_abstract(item),
                    'authors': [authorship.get('author', {}).get('display_name', '') 
                               for authorship in item.get('authorships', []) 
                               if authorship.get('author', {}).get('display_name')],
                    'year': item.get('publication_year'),
                    'doi': item.get('doi', '').replace('https://doi.org/', '') if item.get('doi') else None,
                    'url': primary_location.get('landing_page_url'),
                    'open_access': item.get('open_access', {}).get('is_oa', False),
                    'citation_count': item.get('cited_by_count', 0) or 0,
                    'journal': source.get('display_name')
                }
                if paper['title']:
                    papers.append(paper)
            
            self.cache_response(query, 'openalex', papers)
            search_progress.update_status('openalex', 'completed', len(papers))
            return papers
            
        except Exception as e:
            print(f"OpenAlex API error: {e}")
            search_progress.update_status('openalex', 'failed', 0)
            return []
    
    def _get_openalex_abstract(self, item: Dict) -> str:
        """Reconstruct abstract from OpenAlex's inverted index format"""
        abstract_inverted = item.get('abstract_inverted_index', {})
        if not abstract_inverted:
            return ''
        
        try:
            word_positions = []
            for word, positions in abstract_inverted.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort()
            return ' '.join(word for _, word in word_positions)
        except:
            return ''
    
    async def fetch_doaj(self, query: str, client: httpx.AsyncClient, max_results: int = None) -> List[Dict]:
        """DOAJ API - Directory of Open Access Journals (async)"""
        max_results = max_results or Config.MAX_PAPERS_PER_API
        
        cached = self.load_cached(query, 'doaj')
        if cached:
            search_progress.update_status('doaj', 'cached', len(cached))
            return cached
        
        search_progress.update_status('doaj', 'searching')
        
        try:
            params = {
                'q': query,
                'pageSize': max_results
            }
            
            response = await client.get(self.doaj_api_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            for item in data.get('results', []):
                bibjson = item.get('bibjson', {})
                identifiers = bibjson.get('identifier', [])
                links = bibjson.get('link', [])
                
                paper = {
                    'source': 'doaj',
                    'title': bibjson.get('title', ''),
                    'abstract': bibjson.get('abstract', '') or '',
                    'authors': [author.get('name', '') for author in bibjson.get('author', []) if author.get('name')],
                    'year': int(bibjson.get('year')) if bibjson.get('year') else None,
                    'doi': identifiers[0].get('id') if identifiers else None,
                    'journal': bibjson.get('journal', {}).get('title'),
                    'url': links[0].get('url') if links else None,
                    'keywords': bibjson.get('keywords', []),
                    'citation_count': 0
                }
                if paper['title']:
                    papers.append(paper)
            
            self.cache_response(query, 'doaj', papers)
            search_progress.update_status('doaj', 'completed', len(papers))
            return papers
            
        except Exception as e:
            print(f"DOAJ API error: {e}")
            search_progress.update_status('doaj', 'failed', 0)
            return []
    
    async def fetch_europe_pmc(self, query: str, client: httpx.AsyncClient, max_results: int = None) -> List[Dict]:
        """Europe PMC API - Open science platform for life sciences (async)"""
        max_results = max_results or Config.MAX_PAPERS_PER_API
        
        cached = self.load_cached(query, 'europe_pmc')
        if cached:
            search_progress.update_status('europe_pmc', 'cached', len(cached))
            return cached
        
        search_progress.update_status('europe_pmc', 'searching')
        
        try:
            params = {
                'query': query,
                'format': 'json',
                'pageSize': max_results,
                'resultType': 'core'
            }
            
            response = await client.get(self.europe_pmc_api_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            for item in data.get('resultList', {}).get('result', []):
                authors = item.get('authorString', '').split(', ') if item.get('authorString') else []
                
                paper = {
                    'source': 'europe_pmc',
                    'title': item.get('title', ''),
                    'abstract': item.get('abstractText', '') or '',
                    'authors': [a.strip() for a in authors if a.strip()],
                    'year': int(item.get('pubYear')) if item.get('pubYear') else None,
                    'doi': item.get('doi'),
                    'pmid': item.get('pmid'),
                    'pmcid': item.get('pmcid'),
                    'journal': item.get('journalTitle'),
                    'url': f"https://europepmc.org/article/{item.get('source', '')}/{item.get('id', '')}",
                    'citation_count': item.get('citedByCount', 0) or 0
                }
                if paper['title']:
                    papers.append(paper)
            
            self.cache_response(query, 'europe_pmc', papers)
            search_progress.update_status('europe_pmc', 'completed', len(papers))
            return papers
            
        except Exception as e:
            print(f"Europe PMC API error: {e}")
            search_progress.update_status('europe_pmc', 'failed', 0)
            return []
    
    # ==================== ASYNC PARALLEL FETCH METHOD ====================
    
    async def fetch_papers_parallel(self, query: str, apis: List[str] = None) -> List[Dict]:
        """
        ⚡ ASYNC PARALLEL API FETCHING - Core Performance Optimization
        Uses asyncio.gather() with per-API timeouts for true concurrent I/O.
        Worst-case wall time = max(individual timeouts) ≈ 10 seconds.
        """
        # Normalize query for consistent cache keys across all APIs
        query = query.lower().strip()
        
        if apis is None:
            apis = [
                'arxiv', 'pubmed', 'semantic_scholar', 'core',
                'crossref', 'openalex', 'doaj', 'europe_pmc'
            ]
        
        all_papers = []
        
        # Start progress tracking
        search_progress.start()
        
        print(f"\n{'='*60}")
        print(f"[>>] Starting ASYNC PARALLEL search for: '{query}'")
        print(f"   Fetching from {len(apis)} APIs simultaneously...")
        print(f"   Per-API timeouts: {', '.join(f'{a}={API_TIMEOUTS[a]}s' for a in apis if a in API_TIMEOUTS)}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Create a single shared httpx async client for all HTTP-based APIs
        async with httpx.AsyncClient(
            headers=self._async_headers,
            follow_redirects=True,
            timeout=httpx.Timeout(30.0)  # Safety net; per-API timeouts via wait_for control actual limits
        ) as client:
            
            async def safe_fetch(api_name: str, coro) -> List[Dict]:
                """Wrap a fetch coroutine with per-API timeout and error handling"""
                api_start = time.time()
                try:
                    result = await asyncio.wait_for(coro, timeout=API_TIMEOUTS.get(api_name, 10))
                    elapsed = time.time() - api_start
                    if result:
                        print(f"  [+] {self.api_display_names.get(api_name, api_name)}: "
                              f"{len(result)} papers ({elapsed:.1f}s)")
                    else:
                        print(f"  [!] {self.api_display_names.get(api_name, api_name)}: "
                              f"0 papers ({elapsed:.1f}s)")
                    return result or []
                except asyncio.TimeoutError:
                    elapsed = time.time() - api_start
                    print(f"  [TIMEOUT] {self.api_display_names.get(api_name, api_name)}: "
                          f"timed out after {elapsed:.1f}s (limit: {API_TIMEOUTS.get(api_name, 10)}s)")
                    search_progress.update_status(api_name, 'timeout', 0)
                    return []
                except Exception as e:
                    elapsed = time.time() - api_start
                    print(f"  [X] {self.api_display_names.get(api_name, api_name)}: "
                          f"Error - {e} ({elapsed:.1f}s)")
                    search_progress.update_status(api_name, 'failed', 0)
                    return []
            
            # Build coroutine map — sync libs wrapped in to_thread, HTTP APIs use async httpx
            coro_map = {
                'arxiv': safe_fetch('arxiv', asyncio.to_thread(self.fetch_arxiv, query)),
                'pubmed': safe_fetch('pubmed', asyncio.to_thread(self.fetch_pubmed, query)),
                'semantic_scholar': safe_fetch('semantic_scholar', self.fetch_semantic_scholar(query, client)),
                'core': safe_fetch('core', self.fetch_core(query, client)),
                'crossref': safe_fetch('crossref', self.fetch_crossref(query, client)),
                'openalex': safe_fetch('openalex', self.fetch_openalex(query, client)),
                'doaj': safe_fetch('doaj', self.fetch_doaj(query, client)),
                'europe_pmc': safe_fetch('europe_pmc', self.fetch_europe_pmc(query, client)),
            }
            
            # Only gather requested APIs — all fire at the exact same time
            tasks = [coro_map[api] for api in apis if api in coro_map]
            results = await asyncio.gather(*tasks)
            
            # Collect all papers from completed APIs
            for result in results:
                if isinstance(result, list):
                    all_papers.extend(result)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"[TIME] Async parallel fetch completed in {elapsed:.2f} seconds")
        print(f"   Total papers before dedup: {len(all_papers)}")
        
        # Deduplicate papers using advanced matching
        unique_papers = self._deduplicate_papers(all_papers)
        
        print(f"   Unique papers after dedup: {len(unique_papers)}")
        print(f"{'='*60}\n")
        
        return unique_papers[:Config.MAX_TOTAL_PAPERS]
    
    def fetch_papers_parallel_sync(self, query: str, apis: List[str] = None) -> List[Dict]:
        """
        Sync wrapper for Flask compatibility.
        Bridges async fetch_papers_parallel into sync world via asyncio.run().
        Safe to call from Flask request threads (threaded=True).
        """
        return asyncio.run(self.fetch_papers_parallel(query, apis))
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Enhanced deduplication with DOI and title matching
        Prioritizes papers with more metadata (citations, abstracts)
        """
        seen_dois = set()
        seen_titles = set()
        unique_papers = []
        
        # Sort by citation count (desc) to keep most cited version
        sorted_papers = sorted(papers, key=lambda x: x.get('citation_count', 0) or 0, reverse=True)
        
        for paper in sorted_papers:
            # Skip if no title
            if not paper.get('title'):
                continue
            
            # Normalize title for comparison
            title_normalized = paper['title'].lower().strip()
            title_normalized = ''.join(c for c in title_normalized if c.isalnum() or c.isspace())
            
            # Check DOI first (most reliable)
            doi = paper.get('doi')
            if doi:
                doi_normalized = doi.lower().strip()
                if doi_normalized in seen_dois:
                    continue
                seen_dois.add(doi_normalized)
            
            # Check title similarity
            if title_normalized in seen_titles:
                continue
            
            seen_titles.add(title_normalized)
            unique_papers.append(paper)
        
        return unique_papers
    
    # Legacy method for backward compatibility
    def fetch_papers(self, query: str, apis: List[str] = None) -> List[Dict]:
        """Backward compatible wrapper - uses async parallel fetching"""
        return self.fetch_papers_parallel_sync(query, apis)


# Global instance
api_manager = APIManager()