# database.py
"""
Enhanced Database Manager with:
- Paper storage and retrieval
- Citation tracking
- Query analytics
- Backward compatibility
"""

import sqlite3
import json
from datetime import datetime
from config import Config
from typing import Dict, Any, List, Optional


class PaperManager:
    """Enhanced paper database manager"""
    
    def __init__(self):
        self.conn = sqlite3.connect(Config.SQLITE_DB_PATH, check_same_thread=False)
        self.create_tables()
        self._migrate_schema()
    
    def _migrate_schema(self):
        """Migrate database schema to add missing columns for backward compatibility"""
        cursor = self.conn.cursor()
        
        # Check and add missing columns to papers table
        cursor.execute("PRAGMA table_info(papers)")
        papers_columns = {row[1] for row in cursor.fetchall()}
        
        if 'keywords' not in papers_columns:
            try:
                cursor.execute('ALTER TABLE papers ADD COLUMN keywords TEXT')
                print("✓ Added 'keywords' column to papers table")
            except sqlite3.OperationalError:
                pass  # Column already exists
        
        if 'embedding_id' not in papers_columns:
            try:
                cursor.execute('ALTER TABLE papers ADD COLUMN embedding_id INTEGER')
                print("✓ Added 'embedding_id' column to papers table")
            except sqlite3.OperationalError:
                pass
        
        # Check and add missing columns to queries table
        cursor.execute("PRAGMA table_info(queries)")
        queries_columns = {row[1] for row in cursor.fetchall()}
        
        if 'search_time' not in queries_columns:
            try:
                cursor.execute('ALTER TABLE queries ADD COLUMN search_time REAL')
                print("✓ Added 'search_time' column to queries table")
            except sqlite3.OperationalError:
                pass
        
        self.conn.commit()
    
    def create_tables(self):
        """Create database tables with enhanced schema"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT,
                authors TEXT,
                year INTEGER,
                doi TEXT,
                journal TEXT,
                url TEXT,
                citation_count INTEGER DEFAULT 0,
                embedding_id INTEGER,
                keywords TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(title, source)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                feature TEXT NOT NULL,
                result_count INTEGER,
                search_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster title lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_papers_title 
            ON papers(title)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_papers_doi 
            ON papers(doi)
        ''')
        
        self.conn.commit()
    
    def save_paper(self, paper_data: Dict[str, Any]) -> Optional[int]:
        """Save paper to database and return ID"""
        cursor = self.conn.cursor()
        
        try:
            # Handle keywords if present
            keywords = paper_data.get('keywords', [])
            if isinstance(keywords, list):
                keywords = json.dumps(keywords)
            
            cursor.execute('''
                INSERT OR IGNORE INTO papers 
                (source, title, abstract, authors, year, doi, journal, url, citation_count, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                paper_data.get('source', ''),
                paper_data.get('title', ''),
                paper_data.get('abstract', ''),
                json.dumps(paper_data.get('authors', [])),
                paper_data.get('year'),
                paper_data.get('doi', ''),
                paper_data.get('journal', ''),
                paper_data.get('url', ''),
                paper_data.get('citation_count', 0),
                keywords
            ))
            
            if cursor.rowcount > 0:
                paper_id = cursor.lastrowid
            else:
                # Paper exists, get its ID
                cursor.execute('SELECT id FROM papers WHERE title = ? AND source = ?', 
                             (paper_data['title'], paper_data['source']))
                result = cursor.fetchone()
                paper_id = result[0] if result else None
            
            self.conn.commit()
            return paper_id
            
        except Exception as e:
            print(f"Error saving paper: {e}")
            self.conn.rollback()
            return None
    
    def get_paper_by_index(self, index: int) -> Dict[str, Any]:
        """Get paper by its index (for FAISS retrieval)"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT * FROM papers WHERE id = ?
        ''', (index + 1,))  # FAISS indices are 0-based, DB IDs are 1-based
        
        row = cursor.fetchone()
        if row:
            return self._row_to_dict(row)
        return {}
    
    def get_paper_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Get paper by title"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM papers WHERE title = ?', (title,))
        row = cursor.fetchone()
        if row:
            return self._row_to_dict(row)
        return None
    
    def get_paper_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """Get paper by DOI"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM papers WHERE doi = ?', (doi,))
        row = cursor.fetchone()
        if row:
            return self._row_to_dict(row)
        return None
    
    def search_papers(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search papers by title or abstract"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM papers 
            WHERE title LIKE ? OR abstract LIKE ?
            ORDER BY citation_count DESC
            LIMIT ?
        ''', (f'%{query}%', f'%{query}%', limit))
        
        return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert database row to dictionary"""
        try:
            authors = json.loads(row[4]) if row[4] else []
        except:
            authors = []
        
        try:
            keywords = json.loads(row[11]) if len(row) > 11 and row[11] else []
        except:
            keywords = []
        
        return {
            'id': row[0],
            'source': row[1],
            'title': row[2],
            'abstract': row[3],
            'authors': authors,
            'year': row[5],
            'doi': row[6],
            'journal': row[7],
            'url': row[8],
            'citation_count': row[9] or 0,
            'embedding_id': row[10],
            'keywords': keywords,
            'created_at': row[12] if len(row) > 12 else None
        }
    
    def log_query(self, query_text: str, feature: str, result_count: int, search_time: float = 0):
        """Log user queries for analytics"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO queries (query_text, feature, result_count, search_time)
            VALUES (?, ?, ?, ?)
        ''', (query_text, feature, result_count, search_time))
        self.conn.commit()
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict]:
        """Get recent queries for analytics"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT query_text, feature, result_count, search_time, created_at
            FROM queries
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        return [
            {
                'query': row[0],
                'feature': row[1],
                'results': row[2],
                'time': row[3],
                'timestamp': row[4]
            }
            for row in cursor.fetchall()
        ]
    
    def get_paper_count(self) -> int:
        """Get total number of indexed papers"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM papers')
        return cursor.fetchone()[0]
    
    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()


class CitationGenerator:
    """
    Automatic Citation Generator supporting:
    - APA 7th Edition
    - MLA 9th Edition
    - BibTeX
    - Chicago
    """
    
    @staticmethod
    def format_authors_apa(authors: List[str]) -> str:
        """Format authors in APA style"""
        if not authors:
            return "Unknown Author"
        
        formatted = []
        for author in authors[:20]:  # APA shows up to 20 authors
            parts = author.strip().split()
            if len(parts) >= 2:
                # Last, F. M. format
                last = parts[-1]
                initials = ' '.join([f"{p[0]}." for p in parts[:-1] if p])
                formatted.append(f"{last}, {initials}")
            else:
                formatted.append(author)
        
        if len(formatted) == 1:
            return formatted[0]
        elif len(formatted) == 2:
            return f"{formatted[0]} & {formatted[1]}"
        elif len(formatted) <= 20:
            return ', '.join(formatted[:-1]) + f", & {formatted[-1]}"
        else:
            return ', '.join(formatted[:19]) + f", ... {formatted[-1]}"
    
    @staticmethod
    def format_authors_mla(authors: List[str]) -> str:
        """Format authors in MLA style"""
        if not authors:
            return "Unknown Author"
        
        if len(authors) == 1:
            parts = authors[0].strip().split()
            if len(parts) >= 2:
                return f"{parts[-1]}, {' '.join(parts[:-1])}"
            return authors[0]
        elif len(authors) == 2:
            first = authors[0].strip().split()
            first_formatted = f"{first[-1]}, {' '.join(first[:-1])}" if len(first) >= 2 else authors[0]
            return f"{first_formatted}, and {authors[1]}"
        else:
            first = authors[0].strip().split()
            first_formatted = f"{first[-1]}, {' '.join(first[:-1])}" if len(first) >= 2 else authors[0]
            return f"{first_formatted}, et al."
    
    @staticmethod
    def format_authors_bibtex(authors: List[str]) -> str:
        """Format authors for BibTeX"""
        return " and ".join(authors) if authors else "Unknown Author"
    
    @classmethod
    def generate_apa(cls, paper: Dict[str, Any]) -> str:
        """Generate APA 7th edition citation"""
        authors = cls.format_authors_apa(paper.get('authors', []))
        year = paper.get('year', 'n.d.')
        title = paper.get('title', 'Untitled')
        journal = paper.get('journal', '')
        doi = paper.get('doi', '')
        
        citation = f"{authors} ({year}). {title}."
        
        if journal:
            citation += f" *{journal}*."
        
        if doi:
            citation += f" https://doi.org/{doi}"
        
        return citation
    
    @classmethod
    def generate_mla(cls, paper: Dict[str, Any]) -> str:
        """Generate MLA 9th edition citation"""
        authors = cls.format_authors_mla(paper.get('authors', []))
        title = paper.get('title', 'Untitled')
        journal = paper.get('journal', '')
        year = paper.get('year', '')
        doi = paper.get('doi', '')
        url = paper.get('url', '')
        
        citation = f'{authors}. "{title}."'
        
        if journal:
            citation += f" *{journal}*"
        
        if year:
            citation += f", {year}"
        
        citation += "."
        
        if doi:
            citation += f" doi:{doi}."
        elif url:
            citation += f" {url}."
        
        return citation
    
    @classmethod
    def generate_bibtex(cls, paper: Dict[str, Any]) -> str:
        """Generate BibTeX entry"""
        # Create citation key from first author and year
        authors = paper.get('authors', ['unknown'])
        first_author = authors[0].split()[-1].lower() if authors else 'unknown'
        year = paper.get('year', 'nd')
        title_words = paper.get('title', 'untitled').split()[:2]
        key = f"{first_author}{year}{''.join(w.lower() for w in title_words)}"
        key = ''.join(c for c in key if c.isalnum())
        
        bibtex = f"@article{{{key},\n"
        bibtex += f'  author = {{{cls.format_authors_bibtex(paper.get("authors", []))}}},\n'
        bibtex += f'  title = {{{paper.get("title", "Untitled")}}},\n'
        
        if paper.get('journal'):
            bibtex += f'  journal = {{{paper.get("journal")}}},\n'
        
        if paper.get('year'):
            bibtex += f'  year = {{{paper.get("year")}}},\n'
        
        if paper.get('doi'):
            bibtex += f'  doi = {{{paper.get("doi")}}},\n'
        
        if paper.get('url'):
            bibtex += f'  url = {{{paper.get("url")}}},\n'
        
        bibtex = bibtex.rstrip(',\n') + '\n}'
        return bibtex
    
    @classmethod
    def generate_chicago(cls, paper: Dict[str, Any]) -> str:
        """Generate Chicago style citation"""
        authors = paper.get('authors', ['Unknown Author'])
        
        # Format first author as Last, First
        if authors:
            first = authors[0].strip().split()
            if len(first) >= 2:
                author_str = f"{first[-1]}, {' '.join(first[:-1])}"
            else:
                author_str = authors[0]
            
            if len(authors) > 1:
                author_str += ", " + ", ".join(authors[1:])
        else:
            author_str = "Unknown Author"
        
        title = paper.get('title', 'Untitled')
        journal = paper.get('journal', '')
        year = paper.get('year', '')
        doi = paper.get('doi', '')
        
        citation = f'{author_str}. "{title}."'
        
        if journal:
            citation += f" *{journal}*"
        
        if year:
            citation += f" ({year})"
        
        citation += "."
        
        if doi:
            citation += f" https://doi.org/{doi}."
        
        return citation
    
    @classmethod
    def generate_all(cls, paper: Dict[str, Any]) -> Dict[str, str]:
        """Generate citations in all formats"""
        return {
            'apa': cls.generate_apa(paper),
            'mla': cls.generate_mla(paper),
            'bibtex': cls.generate_bibtex(paper),
            'chicago': cls.generate_chicago(paper)
        }


# Global instances
paper_manager = PaperManager()
citation_generator = CitationGenerator()