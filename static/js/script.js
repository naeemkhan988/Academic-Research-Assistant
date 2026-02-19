// static/js/script.js
/**
 * AI Academic Research Assistant - Premium JavaScript
 * Features:
 * - Live Search Progress (SSE)
 * - Streaming AI Responses
 * - Citation Generation
 * - Interactive UI Enhancements
 */

document.addEventListener('DOMContentLoaded', function () {
    initializeSearch();
    initializeTabs();
    initializeCitations();
    initializeCopyButtons();
    initializeTooltips();
});


// ==================== SEARCH FUNCTIONALITY ====================

function initializeSearch() {
    const researchForm = document.getElementById('researchForm');
    const submitBtn = document.getElementById('submitBtn');
    const searchProgress = document.getElementById('searchProgress');

    if (!researchForm) return;

    researchForm.addEventListener('submit', function (e) {
        // Show loading state
        showLoadingState(submitBtn);

        // Show progress tracker
        if (searchProgress) {
            searchProgress.style.display = 'block';
            startProgressTimer();
            initializeApiStatusGrid();
        }
    });
}

function showLoadingState(btn) {
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');

    if (btnText && btnLoading) {
        btnText.style.display = 'none';
        btnLoading.style.display = 'flex';
        btn.disabled = true;
    }

    // Safety timeout - re-enable after 60s
    setTimeout(() => {
        if (btnText && btnLoading) {
            btnText.style.display = 'flex';
            btnLoading.style.display = 'none';
            btn.disabled = false;
        }
    }, 60000);
}

function startProgressTimer() {
    const timerEl = document.getElementById('progressTimer');
    if (!timerEl) return;

    let startTime = Date.now();

    const interval = setInterval(() => {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        timerEl.textContent = `${elapsed}s`;

        // Stop after 60 seconds
        if (elapsed > 60) {
            clearInterval(interval);
        }
    }, 100);

    // Store interval for cleanup
    window.progressTimerInterval = interval;
}

function initializeApiStatusGrid() {
    const grid = document.getElementById('apiStatusGrid');
    if (!grid) return;

    const apis = [
        { key: 'arxiv', name: '📚 arXiv' },
        { key: 'pubmed', name: '🏥 PubMed' },
        { key: 'semantic_scholar', name: '🔬 Semantic Scholar' },
        { key: 'core', name: '📖 CORE' },
        { key: 'crossref', name: '📄 CrossRef' },
        { key: 'openalex', name: '🌐 OpenAlex' },
        { key: 'doaj', name: '📕 DOAJ' },
        { key: 'europe_pmc', name: '🧬 Europe PMC' }
    ];

    grid.innerHTML = apis.map(api => `
        <div class="api-status-item searching" id="status-${api.key}">
            <i class="fas fa-spinner fa-spin"></i>
            <span>${api.name}</span>
            <span class="api-count"></span>
        </div>
    `).join('');
}

function updateApiStatus(apiKey, status, count = 0) {
    const statusEl = document.getElementById(`status-${apiKey}`);
    if (!statusEl) return;

    statusEl.classList.remove('searching', 'completed', 'failed', 'cached');

    let icon, countText;

    switch (status) {
        case 'searching':
            statusEl.classList.add('searching');
            icon = '<i class="fas fa-spinner fa-spin"></i>';
            countText = '';
            break;
        case 'completed':
            statusEl.classList.add('completed');
            icon = '<i class="fas fa-check-circle"></i>';
            countText = count > 0 ? `(${count})` : '';
            break;
        case 'cached':
            statusEl.classList.add('completed');
            icon = '<i class="fas fa-database"></i>';
            countText = count > 0 ? `(${count})` : '';
            break;
        case 'failed':
            statusEl.classList.add('failed');
            icon = '<i class="fas fa-times-circle"></i>';
            countText = '';
            break;
        default:
            icon = '<i class="fas fa-circle"></i>';
            countText = '';
    }

    const apiName = statusEl.querySelector('span:first-of-type').textContent;
    const countEl = statusEl.querySelector('.api-count');

    statusEl.innerHTML = `${icon}<span>${apiName}</span><span class="api-count">${countText}</span>`;
}


// ==================== STREAMING AI RESPONSE ====================

function initializeStreamingResponse(resultId) {
    const analysisContent = document.getElementById('analysisContent');
    if (!analysisContent || !resultId) return;

    // Clear existing content
    analysisContent.innerHTML = '<div class="streaming-cursor">▊</div>';

    // Accumulate raw markdown text for proper rendering
    let markdownBuffer = '';

    const eventSource = new EventSource(`/api/analyze/stream/${resultId}`);

    eventSource.onmessage = function (event) {
        const data = JSON.parse(event.data);

        switch (data.type) {
            case 'start':
                markdownBuffer = '';
                analysisContent.innerHTML = '';
                break;

            case 'token':
                // Accumulate raw markdown
                markdownBuffer += data.content;

                // Render full markdown buffer as HTML using marked.js
                if (typeof marked !== 'undefined') {
                    analysisContent.innerHTML = marked.parse(markdownBuffer) +
                        '<span class="streaming-cursor">▊</span>';
                } else {
                    analysisContent.innerHTML = markdownBuffer.replace(/\n/g, '<br>') +
                        '<span class="streaming-cursor">▊</span>';
                }

                // Auto-scroll to bottom
                analysisContent.scrollTop = analysisContent.scrollHeight;
                break;

            case 'complete':
                // Final render — remove cursor and do a clean markdown parse
                if (typeof marked !== 'undefined') {
                    analysisContent.innerHTML = marked.parse(markdownBuffer);
                } else {
                    analysisContent.innerHTML = markdownBuffer.replace(/\n/g, '<br>');
                }
                eventSource.close();
                break;

            case 'error':
                analysisContent.innerHTML += `<div class="alert alert-danger mt-3">${data.message}</div>`;
                eventSource.close();
                break;
        }
    };

    eventSource.onerror = function () {
        eventSource.close();
        // Final render on error
        if (markdownBuffer && typeof marked !== 'undefined') {
            analysisContent.innerHTML = marked.parse(markdownBuffer);
        }
        const cursor = analysisContent.querySelector('.streaming-cursor');
        if (cursor) cursor.remove();
    };
}


// ==================== TABS ====================

function initializeTabs() {
    const tabButtons = document.querySelectorAll('#resultsTab button');

    tabButtons.forEach(button => {
        button.addEventListener('click', function (e) {
            e.preventDefault();

            // Remove active class from all
            tabButtons.forEach(btn => btn.classList.remove('active'));

            // Add active class to clicked
            this.classList.add('active');

            // Show corresponding tab content
            const target = this.getAttribute('data-bs-target');
            document.querySelectorAll('.tab-pane').forEach(pane => {
                pane.classList.remove('show', 'active');
            });

            const targetPane = document.querySelector(target);
            if (targetPane) {
                targetPane.classList.add('show', 'active');
            }
        });
    });
}


// ==================== CITATIONS ====================

function initializeCitations() {
    // Citation format selector
    const formatOptions = document.querySelectorAll('input[name="citationFormat"]');
    formatOptions.forEach(option => {
        option.addEventListener('change', function () {
            const resultId = document.querySelector('[data-result-id]')?.dataset.resultId;
            if (resultId) {
                loadCitations(resultId, this.value);
            }
        });
    });

    // Generate citations button
    const generateBtn = document.getElementById('generateCitationsBtn');
    if (generateBtn) {
        generateBtn.addEventListener('click', function () {
            const resultId = this.dataset.resultId;
            const format = document.querySelector('input[name="citationFormat"]:checked')?.value || 'apa';
            loadCitations(resultId, format);
        });
    }

    // Individual cite buttons
    document.querySelectorAll('.cite-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            const paperIndex = this.dataset.paperIndex;
            const resultId = this.dataset.resultId;
            showCitationModal(resultId, paperIndex);
        });
    });

    // Copy citations button
    const copyBtn = document.getElementById('copyCitationsBtn');
    if (copyBtn) {
        copyBtn.addEventListener('click', function () {
            const content = document.getElementById('citationsOutput')?.textContent;
            if (content) {
                copyToClipboard(content, this);
            }
        });
    }
}

async function loadCitations(resultId, format) {
    const output = document.getElementById('citationsOutput');
    if (!output) return;

    output.innerHTML = '<div class="text-center py-4"><i class="fas fa-spinner fa-spin me-2"></i>Loading citations...</div>';

    try {
        const response = await fetch(`/api/citations/${resultId}?format=${format}`);
        const data = await response.json();

        if (data.success) {
            output.textContent = data.citations.join('\n\n');
        } else {
            output.innerHTML = `<div class="text-danger">${data.error}</div>`;
        }
    } catch (error) {
        output.innerHTML = `<div class="text-danger">Error loading citations: ${error.message}</div>`;
    }
}

async function showCitationModal(resultId, paperIndex) {
    try {
        const response = await fetch(`/api/citation/${resultId}/${paperIndex}`);
        const data = await response.json();

        if (!data.success) {
            console.error('Failed to load citation');
            return;
        }

        const modalContent = document.getElementById('modalCitationContent');
        if (modalContent) {
            modalContent.innerHTML = `
                <h6 class="mb-3">${data.paper_title}</h6>
                
                <div class="citation-format-section mb-3">
                    <label class="form-label text-accent">APA 7th Edition</label>
                    <div class="citation-text">${data.citations.apa}</div>
                    <button class="btn btn-sm btn-outline-accent mt-2 copy-citation-btn" 
                            data-citation="${escapeHtml(data.citations.apa)}">
                        <i class="fas fa-copy me-1"></i>Copy
                    </button>
                </div>
                
                <div class="citation-format-section mb-3">
                    <label class="form-label text-accent">MLA 9th Edition</label>
                    <div class="citation-text">${data.citations.mla}</div>
                    <button class="btn btn-sm btn-outline-accent mt-2 copy-citation-btn" 
                            data-citation="${escapeHtml(data.citations.mla)}">
                        <i class="fas fa-copy me-1"></i>Copy
                    </button>
                </div>
                
                <div class="citation-format-section mb-3">
                    <label class="form-label text-accent">BibTeX</label>
                    <pre class="citation-text bibtex">${data.citations.bibtex}</pre>
                    <button class="btn btn-sm btn-outline-accent mt-2 copy-citation-btn" 
                            data-citation="${escapeHtml(data.citations.bibtex)}">
                        <i class="fas fa-copy me-1"></i>Copy
                    </button>
                </div>
                
                <div class="citation-format-section">
                    <label class="form-label text-accent">Chicago</label>
                    <div class="citation-text">${data.citations.chicago}</div>
                    <button class="btn btn-sm btn-outline-accent mt-2 copy-citation-btn" 
                            data-citation="${escapeHtml(data.citations.chicago)}">
                        <i class="fas fa-copy me-1"></i>Copy
                    </button>
                </div>
            `;

            // Add copy handlers
            modalContent.querySelectorAll('.copy-citation-btn').forEach(btn => {
                btn.addEventListener('click', function () {
                    copyToClipboard(this.dataset.citation, this);
                });
            });
        }

        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('citationModal'));
        modal.show();

    } catch (error) {
        console.error('Citation error:', error);
    }
}


// ==================== COPY FUNCTIONALITY ====================

function initializeCopyButtons() {
    // Copy analysis button
    const copyAnalysisBtn = document.getElementById('copyAnalysisBtn');
    if (copyAnalysisBtn) {
        copyAnalysisBtn.addEventListener('click', function () {
            const content = document.getElementById('analysisContent')?.textContent;
            if (content) {
                copyToClipboard(content, this);
            }
        });
    }
}

function copyToClipboard(text, button) {
    navigator.clipboard.writeText(text).then(() => {
        const originalHtml = button.innerHTML;
        button.innerHTML = '<i class="fas fa-check me-1"></i>Copied!';
        button.classList.add('btn-accent');
        button.classList.remove('btn-outline-accent', 'btn-outline-light');

        setTimeout(() => {
            button.innerHTML = originalHtml;
            button.classList.remove('btn-accent');
            button.classList.add('btn-outline-accent');
        }, 2000);
    }).catch(err => {
        console.error('Copy failed:', err);
    });
}


// ==================== TOOLTIPS ====================

function initializeTooltips() {
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipTriggerList.forEach(el => {
        new bootstrap.Tooltip(el);
    });
}


// ==================== UTILITIES ====================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/"/g, '&quot;');
}

function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}


// ==================== SMOOTH SCROLLING ====================

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});


// ==================== KEYBOARD SHORTCUTS ====================

document.addEventListener('keydown', function (e) {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const form = document.getElementById('researchForm');
        if (form) {
            form.dispatchEvent(new Event('submit'));
        }
    }
});


// ==================== AUTO-RESIZE TEXTAREAS ====================

document.querySelectorAll('textarea').forEach(textarea => {
    textarea.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
});