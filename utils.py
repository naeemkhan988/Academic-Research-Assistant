# utils.py
"""
Enhanced Utilities with:
- Research Gap Analysis Matrix Visualization
- Improved Charts and Graphs
- Enhanced PDF Export with Citations
"""

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from config import Config
from typing import List, Dict, Any, Optional
from collections import Counter


class TextProcessor:
    """Enhanced text processing for research paper analysis"""
    
    def __init__(self):
        self.methodology_keywords = {
            'Machine Learning': ['neural network', 'deep learning', 'random forest', 'svm', 
                               'cnn', 'rnn', 'transformer', 'bert', 'gpt', 'lstm', 'gan'],
            'Statistical Analysis': ['regression', 'anova', 't-test', 'chi-square', 
                                   'correlation', 'factor analysis', 'bayesian'],
            'Qualitative Research': ['interview', 'case study', 'ethnography', 
                                   'phenomenology', 'content analysis', 'thematic'],
            'Experimental': ['randomized controlled trial', 'rct', 'experiment', 
                            'control group', 'randomized', 'clinical trial'],
            'Survey Research': ['questionnaire', 'survey', 'likert', 'cross-sectional', 
                              'longitudinal'],
            'Simulation': ['simulation', 'monte carlo', 'agent-based', 'numerical analysis'],
            'Meta-Analysis': ['meta-analysis', 'systematic review', 'literature review'],
            'Mixed Methods': ['mixed methods', 'qualitative and quantitative', 'triangulation']
        }
        
        self.research_domains = {
            'Healthcare': ['medical', 'clinical', 'patient', 'disease', 'treatment', 'diagnosis'],
            'Computer Science': ['algorithm', 'software', 'computing', 'programming', 'data'],
            'Social Sciences': ['society', 'behavior', 'psychology', 'sociology', 'culture'],
            'Engineering': ['design', 'system', 'mechanical', 'electrical', 'structural'],
            'Natural Sciences': ['biology', 'chemistry', 'physics', 'environment', 'ecology'],
            'Business': ['management', 'marketing', 'finance', 'economics', 'strategy']
        }
    
    def extract_methodology(self, abstract: str) -> str:
        """Extract research methodology from abstract"""
        if not abstract:
            return 'Not specified'
        
        abstract_lower = abstract.lower()
        methodologies = []
        
        for method_type, keywords in self.methodology_keywords.items():
            for keyword in keywords:
                if keyword in abstract_lower:
                    methodologies.append(method_type)
                    break
        
        return ', '.join(set(methodologies)) if methodologies else 'Not specified'
    
    def extract_domain(self, text: str) -> str:
        """Extract research domain from text"""
        if not text:
            return 'General'
        
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.research_domains.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return 'General'
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        return text


class ResearchGapAnalyzer:
    """
    Research Gap Analysis Matrix Generator
    Creates visual comparison of topics, methods, and missing areas
    """
    
    def __init__(self):
        self.text_processor = TextProcessor()
    
    def analyze_gaps(self, papers: List[Dict]) -> Dict[str, Any]:
        """Analyze research gaps across papers"""
        if not papers:
            return {}
        
        # Extract metadata
        years = []
        methods = []
        domains = []
        topics = []
        
        for paper in papers:
            # Year
            if paper.get('year'):
                years.append(int(paper['year']))
            
            # Methodology
            abstract = paper.get('abstract', '')
            method = self.text_processor.extract_methodology(abstract)
            if method != 'Not specified':
                methods.extend([m.strip() for m in method.split(',')])
            
            # Domain
            text = f"{paper.get('title', '')} {abstract}"
            domain = self.text_processor.extract_domain(text)
            domains.append(domain)
            
            # Extract key topics from title
            title_words = paper.get('title', '').lower().split()
            topics.extend([w for w in title_words if len(w) > 5])
        
        return {
            'year_distribution': Counter(years),
            'method_distribution': Counter(methods),
            'domain_distribution': Counter(domains),
            'topic_frequency': Counter(topics).most_common(20),
            'total_papers': len(papers),
            'year_range': (min(years), max(years)) if years else (None, None),
            'identified_gaps': self._identify_gaps(methods, domains, years)
        }
    
    def _identify_gaps(self, methods: List, domains: List, years: List) -> List[str]:
        """Identify potential research gaps"""
        gaps = []
        
        method_counts = Counter(methods)
        domain_counts = Counter(domains)
        
        # Check for underrepresented methods
        all_methods = set(self.text_processor.methodology_keywords.keys())
        used_methods = set(method_counts.keys())
        missing_methods = all_methods - used_methods
        
        if missing_methods:
            gaps.append(f"Underutilized methodologies: {', '.join(list(missing_methods)[:3])}")
        
        # Check for temporal gaps
        if years:
            year_range = range(min(years), max(years) + 1)
            missing_years = set(year_range) - set(years)
            if len(missing_years) > 2:
                gaps.append(f"Limited research in years: {', '.join(map(str, sorted(missing_years)[:3]))}")
        
        # Check for method diversity
        if len(method_counts) < 3:
            gaps.append("Limited methodological diversity - consider mixed methods approaches")
        
        return gaps
    
    def create_gap_matrix_plot(self, papers: List[Dict], query: str) -> Optional[str]:
        """Create Research Gap Analysis Matrix visualization"""
        if not papers or len(papers) < 3:
            return None
        
        analysis = self.analyze_gaps(papers)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Research Gap Analysis: {query}', fontsize=14, fontweight='bold', color='white')
        
        # Style
        plt.style.use('dark_background')
        colors_palette = ['#00b3a8', '#00d4c8', '#00968c', '#007a72', '#005e58', '#004d47']
        
        # 1. Methodology Distribution (Top Left)
        ax1 = axes[0, 0]
        methods = analysis['method_distribution']
        if methods:
            method_names = list(methods.keys())[:6]
            method_counts = [methods[m] for m in method_names]
            bars = ax1.barh(method_names, method_counts, color=colors_palette[:len(method_names)])
            ax1.set_xlabel('Count')
            ax1.set_title('🧪 Methodology Distribution', fontweight='bold')
            for bar, count in zip(bars, method_counts):
                ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                        str(count), va='center', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No methodology data', ha='center', va='center')
            ax1.set_title('🧪 Methodology Distribution', fontweight='bold')
        
        # 2. Year Distribution (Top Right)
        ax2 = axes[0, 1]
        years = analysis['year_distribution']
        if years:
            year_keys = sorted(years.keys())
            year_values = [years[y] for y in year_keys]
            ax2.fill_between(year_keys, year_values, alpha=0.3, color='#00b3a8')
            ax2.plot(year_keys, year_values, color='#00d4c8', linewidth=2, marker='o')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Publications')
            ax2.set_title('📈 Publication Timeline', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No year data', ha='center', va='center')
            ax2.set_title('📈 Publication Timeline', fontweight='bold')
        
        # 3. Domain Distribution (Bottom Left)
        ax3 = axes[1, 0]
        domains = analysis['domain_distribution']
        if domains:
            domain_names = list(domains.keys())[:6]
            domain_counts = [domains[d] for d in domain_names]
            wedges, texts, autotexts = ax3.pie(domain_counts, labels=domain_names, 
                                               autopct='%1.1f%%', colors=colors_palette[:len(domain_names)])
            ax3.set_title('🎯 Research Domains', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No domain data', ha='center', va='center')
            ax3.set_title('🎯 Research Domains', fontweight='bold')
        
        # 4. Gap Analysis Summary (Bottom Right)
        ax4 = axes[1, 1]
        ax4.axis('off')
        gaps = analysis.get('identified_gaps', [])
        
        gap_text = f"📊 Analysis Summary\n\n"
        gap_text += f"Total Papers: {analysis['total_papers']}\n"
        year_range = analysis.get('year_range', (None, None))
        if year_range[0]:
            gap_text += f"Year Range: {year_range[0]} - {year_range[1]}\n"
        gap_text += f"Methods Found: {len(analysis['method_distribution'])}\n"
        gap_text += f"Domains Covered: {len(analysis['domain_distribution'])}\n\n"
        
        gap_text += "🔍 Identified Gaps:\n"
        for gap in gaps[:3]:
            gap_text += f"• {gap}\n"
        
        ax4.text(0.1, 0.9, gap_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gap_analysis_{query.replace(' ', '_')[:30]}_{timestamp}.png"
        plot_path = os.path.join('static/images', filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close()
        
        return f"images/{filename}"


class VisualizationEngine:
    """Enhanced visualization engine with premium charts"""
    
    def __init__(self):
        plt.style.use('dark_background')
        self.colors = ['#00b3a8', '#00d4c8', '#00968c', '#007a72', '#005e58', 
                      '#00ffea', '#00e5d4', '#00ccbd', '#00b3a8', '#009991']
        self.gap_analyzer = ResearchGapAnalyzer()
    
    def create_trends_plot(self, papers: List[Dict], query: str) -> Optional[str]:
        """Create enhanced publication trends visualization"""
        if not papers:
            return None
        
        years = []
        for paper in papers:
            if paper.get('year'):
                try:
                    year = int(paper['year'])
                    if 1990 <= year <= 2030:
                        years.append(year)
                except (ValueError, TypeError):
                    continue
        
        if not years or len(years) < 3:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0a0a0a')
        
        # Year distribution with gradient effect
        year_counts = Counter(years)
        sorted_years = sorted(year_counts.keys())
        counts = [year_counts[y] for y in sorted_years]
        
        bars = ax1.bar(sorted_years, counts, color=self.colors[0], alpha=0.8, edgecolor='white')
        
        # Add gradient effect
        for bar in bars:
            bar.set_edgecolor('white')
            bar.set_linewidth(0.5)
        
        ax1.set_xlabel('Publication Year', fontsize=11)
        ax1.set_ylabel('Number of Papers', fontsize=11)
        ax1.set_title(f'📊 Publication Distribution', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.2)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontsize=8)
        
        # Cumulative trend with area fill
        cumulative = np.cumsum(counts)
        ax2.fill_between(sorted_years, cumulative, alpha=0.3, color=self.colors[1])
        ax2.plot(sorted_years, cumulative, color=self.colors[1], linewidth=2.5, marker='o', markersize=4)
        ax2.set_xlabel('Publication Year', fontsize=11)
        ax2.set_ylabel('Cumulative Publications', fontsize=11)
        ax2.set_title('📈 Cumulative Research Growth', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trends_{query.replace(' ', '_')[:30]}_{timestamp}.png"
        plot_path = os.path.join('static/images', filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close()
        
        return f"images/{filename}"
    
    def create_methodology_chart(self, papers: List[Dict]) -> Optional[str]:
        """Create enhanced methodology distribution chart"""
        processor = TextProcessor()
        methodologies = []
        
        for paper in papers:
            method = processor.extract_methodology(paper.get('abstract', ''))
            if method != 'Not specified':
                methodologies.extend([m.strip() for m in method.split(',')])
        
        if not methodologies:
            return None
        
        method_counts = Counter(methodologies)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0a0a0a')
        
        # Bar chart
        methods, counts = zip(*method_counts.most_common(8))
        
        bars = ax1.barh(methods, counts, color=self.colors[:len(methods)])
        ax1.set_xlabel('Frequency', fontsize=11)
        ax1.set_title('🧪 Research Methodologies', fontsize=12, fontweight='bold')
        
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    str(count), ha='left', va='center', fontsize=9)
        
        # Pie chart
        wedges, texts, autotexts = ax2.pie(counts, labels=methods, autopct='%1.1f%%',
                                           colors=self.colors[:len(methods)],
                                           explode=[0.02]*len(methods))
        ax2.set_title('📊 Methodology Distribution', fontsize=12, fontweight='bold')
        
        plt.setp(autotexts, size=8, weight='bold')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"methodology_{timestamp}.png"
        plot_path = os.path.join('static/images', filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close()
        
        return f"images/{filename}"
    
    def create_gap_analysis_matrix(self, papers: List[Dict], query: str) -> Optional[str]:
        """Create Research Gap Analysis Matrix visualization"""
        return self.gap_analyzer.create_gap_matrix_plot(papers, query)
    
    def create_source_distribution(self, papers: List[Dict]) -> Optional[str]:
        """Create source distribution chart"""
        if not papers:
            return None
        
        sources = [p.get('source', 'unknown') for p in papers]
        source_counts = Counter(sources)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#0a0a0a')
        
        source_names = list(source_counts.keys())
        counts = list(source_counts.values())
        
        wedges, texts, autotexts = ax.pie(counts, labels=source_names, autopct='%1.1f%%',
                                          colors=self.colors[:len(source_names)],
                                          explode=[0.02]*len(source_names))
        
        ax.set_title('📚 Data Sources Distribution', fontsize=12, fontweight='bold')
        plt.setp(autotexts, size=9, weight='bold')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sources_{timestamp}.png"
        plot_path = os.path.join('static/images', filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close()
        
        return f"images/{filename}"


class PDFExporter:
    """Enhanced PDF exporter with citations and visualizations"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom styles for PDF export"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#001f3f'),
            spaceAfter=14,
            alignment=1  # Center
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=13,
            textColor=colors.HexColor('#00b3a8'),
            spaceAfter=8,
            spaceBefore=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='Citation',
            parent=self.styles['Normal'],
            fontSize=9,
            leftIndent=20,
            spaceAfter=6,
            textColor=colors.HexColor('#333333')
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubHeading',
            parent=self.styles['Heading3'],
            fontSize=11,
            textColor=colors.HexColor('#00968c'),
            spaceAfter=6,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='BulletItem',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=24,
            spaceAfter=3,
            textColor=colors.HexColor('#333333')
        ))
    
    def _markdown_to_reportlab(self, text: str) -> str:
        """Convert markdown inline formatting to ReportLab-compatible HTML tags.
        
        ReportLab's Paragraph supports a subset of HTML including <b>, <i>, <font>, <br/>.
        This converts markdown syntax to those tags for proper PDF rendering.
        """
        if not text:
            return ""
        
        # Escape XML special characters first (before adding our own HTML tags)
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        # Convert inline formatting (order matters: *** before ** before *)
        # Bold italic: ***text***
        text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
        # Bold: **text**
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        # Italic: *text* (avoid matching lone * used as bullet or multiplication)
        text = re.sub(r'(?<!\*)\*([^\*\n]+?)\*(?!\*)', r'<i>\1</i>', text)
        # Inline code: `text`
        text = re.sub(r'`([^`]+?)`', r'<font face="Courier">\1</font>', text)
        
        # Safety fallback: strip any remaining raw markdown symbols
        text = text.replace('**', '')
        text = text.replace('##', '')
        text = text.replace('# ', '')
        
        return text
    
    def export_to_pdf(self, content: Dict[str, Any], query: str, feature: str) -> str:
        """Export analysis results to enhanced PDF"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_analysis_{query.replace(' ', '_')[:20]}_{timestamp}.pdf"
        pdf_path = os.path.join('static/exports', filename)
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)
        story = []
        
        # Title
        title = Paragraph(f"📚 Research Analysis Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 8))
        
        subtitle = Paragraph(f"<i>Query: {query}</i>", self.styles['Normal'])
        story.append(subtitle)
        story.append(Spacer(1, 16))
        
        # Metadata table
        metadata_data = [
            ['Analysis Feature', feature.title()],
            ['Research Topic', query],
            ['Papers Analyzed', str(content.get('papers_analyzed', 0))],
            ['Generated', datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#00b3a8')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (1, 0), (-1, -1), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc'))
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 24))
        
        # Analysis Results
        results_title = Paragraph("📊 Analysis Results", self.styles['CustomHeading'])
        story.append(results_title)
        
        result_text = content.get('result', 'No results generated.')
        
        # Process markdown-formatted LLM response into styled PDF paragraphs
        result_paragraphs = result_text.split('\n')
        for para in result_paragraphs:
            para = para.strip()
            if not para:
                # Blank line = paragraph break
                story.append(Spacer(1, 6))
                continue
            
            # Horizontal rules (---, ***, ___)
            if re.match(r'^[-*_]{3,}$', para):
                story.append(Spacer(1, 4))
                rule = Table([['']], colWidths=[6*inch], rowHeights=[1])
                rule.setStyle(TableStyle([
                    ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc'))
                ]))
                story.append(rule)
                story.append(Spacer(1, 4))
                continue
            
            # Headers: match # through ###### (regex handles all levels)
            header_match = re.match(r'^(#{1,6})\s+(.*)', para)
            if header_match:
                level = len(header_match.group(1))
                text = self._markdown_to_reportlab(header_match.group(2))
                if level == 1:
                    style = self.styles['Heading2']
                elif level == 2:
                    style = self.styles['CustomHeading']
                else:
                    style = self.styles['SubHeading']
                story.append(Spacer(1, 8 if level <= 2 else 5))
                story.append(Paragraph(text, style))
                story.append(Spacer(1, 3))
                continue
            
            # Bullet points: - item, * item, • item
            if re.match(r'^[-*\u2022]\s+', para):
                bullet_text = re.sub(r'^[-*\u2022]\s+', '', para)
                text = self._markdown_to_reportlab(bullet_text)
                story.append(Paragraph(f'\u2022  {text}', self.styles['BulletItem']))
                story.append(Spacer(1, 2))
                continue
            
            # Numbered lists: 1. item, 2) item, etc.
            num_match = re.match(r'^(\d+[.\)])\s+(.*)', para)
            if num_match:
                num = num_match.group(1)
                text = self._markdown_to_reportlab(num_match.group(2))
                story.append(Paragraph(f'<b>{num}</b>  {text}', self.styles['BulletItem']))
                story.append(Spacer(1, 2))
                continue
            
            # Regular paragraph — convert inline markdown to ReportLab HTML
            text = self._markdown_to_reportlab(para)
            story.append(Paragraph(text, self.styles['Normal']))
            story.append(Spacer(1, 3))
        
        story.append(Spacer(1, 16))
        
        # Top Papers with Citations
        if content.get('top_papers'):
            papers_title = Paragraph("📑 Key Relevant Papers", self.styles['CustomHeading'])
            story.append(papers_title)
            story.append(Spacer(1, 8))
            
            from database import CitationGenerator
            
            for i, paper in enumerate(content['top_papers'][:5], 1):
                # Paper info
                paper_text = f"<b>{i}. {paper.get('title', 'Untitled')}</b><br/>"
                authors = paper.get('authors', [])
                if authors:
                    author_str = ', '.join(authors[:3])
                    if len(authors) > 3:
                        author_str += ' et al.'
                    paper_text += f"<i>Authors:</i> {author_str}<br/>"
                paper_text += f"<i>Year:</i> {paper.get('year', 'N/A')} | "
                paper_text += f"<i>Citations:</i> {paper.get('citation_count', 'N/A')} | "
                paper_text += f"<i>Source:</i> {paper.get('source', 'N/A')}<br/>"
                paper_text += f"<i>Relevance Score:</i> {paper.get('similarity_score', 0):.3f}"
                
                story.append(Paragraph(paper_text, self.styles['Normal']))
                
                # Add APA citation
                apa = CitationGenerator.generate_apa(paper)
                story.append(Paragraph(f"<i>APA Citation:</i> {apa}", self.styles['Citation']))
                story.append(Spacer(1, 10))
        
        # Footer
        story.append(Spacer(1, 20))
        footer = Paragraph(
            "<i>Generated by AI Academic Research Assistant | Powered by GPT-4o-mini</i>",
            ParagraphStyle('Footer', parent=self.styles['Normal'], 
                          fontSize=8, textColor=colors.gray, alignment=1)
        )
        story.append(footer)
        
        doc.build(story)
        return f"exports/{filename}"


# Global instances
text_processor = TextProcessor()
viz_engine = VisualizationEngine()
pdf_exporter = PDFExporter()
gap_analyzer = ResearchGapAnalyzer()