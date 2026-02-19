# app.py
"""
Enhanced Flask Application with:
- Streaming AI Responses (SSE)
- Live Search Progress Updates
- Citation Generation API
- Research Gap Analysis
"""

from flask import Flask, render_template, request, session, send_file, jsonify, redirect, url_for, Response, stream_with_context
import os
import time
import uuid
import json
from config import Config
from api_integrations import api_manager, search_progress
from rag_pipeline import rag_pipeline
from database import PaperManager, CitationGenerator
from utils import viz_engine, pdf_exporter

app = Flask(__name__)
app.config.from_object(Config)

paper_manager = PaperManager()

# In-memory storage for results
results_cache = {}


@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page with search form"""
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        feature = request.form.get('feature', 'summarize')
        
        if not query:
            return render_template('index.html', error="Please enter a research topic.")
        
        try:
            # Fetch papers using parallel API calls
            start_time = time.time()
            papers = api_manager.fetch_papers_parallel(query)
            fetch_time = time.time() - start_time
            
            if not papers:
                return render_template('index.html', 
                                      error="No papers found. Try a different search term.")
            
            # Process with RAG pipeline
            start_time = time.time()
            result = rag_pipeline.process_query(query, feature, papers)
            process_time = time.time() - start_time
            
            # Log query with timing
            paper_manager.log_query(query, feature, len(papers), fetch_time)
            
            # Generate visualizations
            trends_plot = viz_engine.create_trends_plot(papers, query)
            methods_plot = viz_engine.create_methodology_chart(papers)
            gap_plot = viz_engine.create_gap_analysis_matrix(papers, query)
            sources_plot = viz_engine.create_source_distribution(papers)
            
            # Generate unique result ID
            result_id = str(uuid.uuid4())[:8]
            
            # Store results in cache
            results_cache[result_id] = {
                'result': result,
                'query': query,
                'feature': feature,
                'papers': papers,  # Store for citations
                'trends_plot': trends_plot,
                'methods_plot': methods_plot,
                'gap_plot': gap_plot,
                'sources_plot': sources_plot,
                'fetch_time': round(fetch_time, 2),
                'process_time': round(process_time, 2),
                'total_papers': len(papers),
                'result_data': {
                    'result': result.get('result', ''),
                    'papers_analyzed': result.get('papers_analyzed', 0),
                    'top_papers': result.get('top_papers', []),
                    'query': query,
                    'feature': feature
                }
            }
            
            # Store in session
            session['current_result_id'] = result_id
            session['result_data'] = results_cache[result_id]['result_data']
            session.modified = True
            
            return redirect(url_for('results', result_id=result_id))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return render_template('index.html', error=f"An error occurred: {str(e)}")
    
    return render_template('index.html')


@app.route('/results/<result_id>')
def results(result_id):
    """Display results page"""
    if result_id not in results_cache:
        return redirect(url_for('index'))
    
    data = results_cache[result_id]
    
    session['current_result_id'] = result_id
    session['result_data'] = data['result_data']
    session.modified = True
    
    return render_template('results.html',
                          result=data['result'],
                          query=data['query'],
                          feature=data['feature'],
                          trends_plot=data['trends_plot'],
                          methods_plot=data['methods_plot'],
                          gap_plot=data.get('gap_plot'),
                          sources_plot=data.get('sources_plot'),
                          fetch_time=data['fetch_time'],
                          process_time=data['process_time'],
                          total_papers=data['total_papers'],
                          result_id=result_id)


# ==================== STREAMING ENDPOINTS ====================

@app.route('/api/search/stream')
def search_stream():
    """
    Server-Sent Events endpoint for real-time search progress
    Used for live updates: "Searching arXiv...", "Searching PubMed...", etc.
    """
    query = request.args.get('query', '')
    
    def generate():
        # Start search in background
        search_progress.start()
        
        yield f"data: {json.dumps({'type': 'start', 'query': query})}\n\n"
        
        last_status = {}
        max_iterations = 100  # timeout after ~20 seconds
        
        for _ in range(max_iterations):
            current_status = search_progress.get_status()
            
            # Check for changes
            if current_status['apis'] != last_status:
                yield f"data: {json.dumps({'type': 'progress', 'status': current_status})}\n\n"
                last_status = current_status['apis'].copy()
            
            # Check if all APIs have completed
            api_statuses = current_status['apis']
            if len(api_statuses) >= 8:  # All 8 APIs reported
                completed = all(
                    s.get('status') in ['completed', 'failed', 'cached'] 
                    for s in api_statuses.values()
                )
                if completed:
                    yield f"data: {json.dumps({'type': 'complete', 'status': current_status})}\n\n"
                    break
            
            time.sleep(0.2)
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/analyze/stream/<result_id>')
def analyze_stream(result_id):
    """
    Server-Sent Events endpoint for streaming AI response
    Token-by-token output for improved perceived speed
    """
    if result_id not in results_cache:
        return jsonify({'error': 'Result not found'}), 404
    
    data = results_cache[result_id]
    query = data['query']
    feature = data['feature']
    papers = data.get('papers', [])
    
    def generate():
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        
        try:
            # Stream the AI response
            for chunk in rag_pipeline.process_query_stream(query, feature, papers):
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


# ==================== CITATION API ====================

@app.route('/api/citation/<result_id>/<int:paper_index>')
def get_citation(result_id, paper_index):
    """
    Generate citations for a specific paper
    Returns APA, MLA, BibTeX, and Chicago formats
    """
    if result_id not in results_cache:
        return jsonify({'error': 'Result not found'}), 404
    
    top_papers = results_cache[result_id]['result'].get('top_papers', [])
    
    if paper_index < 0 or paper_index >= len(top_papers):
        return jsonify({'error': 'Paper not found'}), 404
    
    paper = top_papers[paper_index]
    citations = CitationGenerator.generate_all(paper)
    
    return jsonify({
        'success': True,
        'paper_title': paper.get('title', ''),
        'citations': citations
    })


@app.route('/api/citations/<result_id>')
def get_all_citations(result_id):
    """Generate citations for all papers in result"""
    if result_id not in results_cache:
        return jsonify({'error': 'Result not found'}), 404
    
    format_type = request.args.get('format', 'apa')
    top_papers = results_cache[result_id]['result'].get('top_papers', [])
    
    citations = []
    for paper in top_papers:
        if format_type == 'apa':
            citations.append(CitationGenerator.generate_apa(paper))
        elif format_type == 'mla':
            citations.append(CitationGenerator.generate_mla(paper))
        elif format_type == 'bibtex':
            citations.append(CitationGenerator.generate_bibtex(paper))
        elif format_type == 'chicago':
            citations.append(CitationGenerator.generate_chicago(paper))
        else:
            citations.append(CitationGenerator.generate_apa(paper))
    
    return jsonify({
        'success': True,
        'format': format_type,
        'count': len(citations),
        'citations': citations
    })


# ==================== EXPORT ENDPOINTS ====================

@app.route('/export')
@app.route('/export/<result_id>')
def export(result_id=None):
    """Export results to PDF"""
    result_data = None
    
    if result_id and result_id in results_cache:
        result_data = results_cache[result_id]['result_data']
    else:
        result_data = session.get('result_data')
    
    if not result_data:
        return "No results to export. Please perform a search first."
    
    try:
        pdf_path = pdf_exporter.export_to_pdf(
            result_data, 
            result_data['query'], 
            result_data['feature']
        )
        return send_file(os.path.join('static', pdf_path), as_attachment=True)
    except Exception as e:
        return f"Error generating PDF: {str(e)}"


@app.route('/export/citations/<result_id>')
def export_citations(result_id):
    """Export all citations in selected format"""
    if result_id not in results_cache:
        return "Result not found", 404
    
    format_type = request.args.get('format', 'bibtex')
    top_papers = results_cache[result_id]['result'].get('top_papers', [])
    
    if format_type == 'bibtex':
        content = '\n\n'.join(CitationGenerator.generate_bibtex(p) for p in top_papers)
        filename = 'citations.bib'
        mimetype = 'application/x-bibtex'
    else:
        citations = []
        for p in top_papers:
            if format_type == 'apa':
                citations.append(CitationGenerator.generate_apa(p))
            elif format_type == 'mla':
                citations.append(CitationGenerator.generate_mla(p))
            else:
                citations.append(CitationGenerator.generate_chicago(p))
        content = '\n\n'.join(citations)
        filename = f'citations_{format_type}.txt'
        mimetype = 'text/plain'
    
    return Response(
        content,
        mimetype=mimetype,
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )


# ==================== STATUS ENDPOINTS ====================

@app.route('/api/status')
def api_status():
    """API status endpoint"""
    from llm_router import llm_router
    router_status = llm_router.get_status()
    
    return jsonify({
        'status': 'operational',
        'version': '3.0',
        'llm_architecture': 'multi-provider-fallback',
        'llm_providers': router_status,
        'apis': [
            'arxiv', 'pubmed', 'semantic_scholar', 'core',
            'crossref', 'openalex', 'doaj', 'europe_pmc'
        ],
        'features': ['summarize', 'gaps', 'compare', 'methods'],
        'capabilities': [
            'parallel_api_fetching',
            'hybrid_search',
            'streaming_responses',
            'citation_generation',
            'gap_analysis_matrix',
            'multi_provider_llm_fallback'
        ],
        'papers_indexed': paper_manager.get_paper_count()
    })


@app.route('/api/recent-queries')
def recent_queries():
    """Get recent search queries"""
    queries = paper_manager.get_recent_queries(10)
    return jsonify({'queries': queries})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)