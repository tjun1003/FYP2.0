/**
 * Adversarial Information Detection System - Frontend Logic
 * Handles API communication, UI updates, and user interactions
 */

const API_URL = 'http://localhost:3000/api/predict';

/**
 * Main prediction function
 * Sends text to API and handles response
 */
async function predict() {
    const text = document.getElementById('textInput').value.trim();
    
    if (!text) {
        showError('Please enter the text to analyze');
        return;
    }

    // Get selected retrieval sources
    const sources = [];
    document.querySelectorAll('.source-checkbox:checked').forEach(cb => {
        sources.push(cb.value);
    });

    // Show loading state
    showLoading('Retrieving web information and analyzing, please wait...');
    hideResult();
    hideError();
    hideRetrievalResults();

    try {
        const requestBody = {
            text: text,
            use_retrieval: true,  // Always use external resources
            sources: sources.length > 0 ? sources : ['wikipedia', 'web']
        };

        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        const data = await response.json();

        if (!response.ok) {
            // Extract detailed error message from response
            const errorDetail = data.detail || data.message || data.error || 'Unknown error occurred';
            throw new Error(errorDetail);
        }

        if (data.error) {
            throw new Error(data.message || data.error);
        }

        // Handle retrieval status
        handleRetrievalStatus(data);

        // Display retrieval results (if any)
        if (data.context && data.context.results && data.context.results.length > 0) {
            displayRetrievalResults(data.context);
        }

        displayResult(data);
    } catch (error) {
        console.error('Error:', error);
        
        // Display user-friendly error message based on error type
        let errorMessage = error.message;
        
        // Categorize error messages for better UX
        if (errorMessage.includes('Failed to fetch') || errorMessage.includes('NetworkError')) {
            errorMessage = '⚠️ Unable to connect to server. Please check if the service is running.';
        } else if (errorMessage.includes('Input validation failed')) {
            errorMessage = '❌ ' + errorMessage.replace('Input validation failed: ', '');
        } else if (errorMessage.includes('Batch input validation failed')) {
            errorMessage = '❌ ' + errorMessage;
        } else if (errorMessage.includes('Model service is not ready')) {
            errorMessage = '⏳ Model is still loading, please wait a moment and try again.';
        } else if (errorMessage.includes('prediction failed')) {
            errorMessage = '⚠️ Prediction failed: ' + errorMessage;
        } else if (errorMessage.includes('timeout')) {
            errorMessage = '⏱️ Request timeout, please try again.';
        } else if (!errorMessage || errorMessage === 'Unknown error occurred') {
            errorMessage = '⚠️ An unexpected error occurred, please try again.';
        } else {
            // Prepend icon for generic errors
            errorMessage = '❌ ' + errorMessage;
        }
        
        showError(errorMessage);
    } finally {
        hideLoading();
    }
}

/**
 * Handle retrieval status messages
 * @param {Object} data - Response data from API
 */
function handleRetrievalStatus(data) {
    const status = data.retrieval_status;
    
    if (status === 'empty') {
        // Retrieval successful but no content found
        showRetrievalWarning('No relevant information found. Judgment will be based solely on the model\'s internal knowledge.');
    } else if (status === 'failed') {
        // Retrieval process failed
        showRetrievalWarning('Retrieval service temporarily unavailable. Judgment will be based solely on the model\'s internal knowledge.', true);
    } else if (status === 'success') {
        // Retrieval successful
        hideRetrievalWarning();
    }
}

/**
 * Display retrieval results from RAG
 * @param {Object} context - Context object containing retrieval results
 */
function displayRetrievalResults(context) {
    const results = context.results || [];
    if (results.length === 0) return;

    const container = document.getElementById('retrievalContent');
    const countEl = document.getElementById('sourceCount');
    
    countEl.textContent = `${results.length} Sources`;
    container.innerHTML = '';

    results.forEach((result, index) => {
        const item = document.createElement('div');
        item.className = 'retrieval-item';
        item.innerHTML = `
            <div class="retrieval-source">
                <span class="source-badge">${result.source}</span>
                <span class="source-number">#${index + 1}</span>
            </div>
            <h4 class="retrieval-title">${escapeHtml(result.title)}</h4>
            <p class="retrieval-content">${escapeHtml(result.content || result.description)}</p>
            <a href="${escapeHtml(result.url)}" target="_blank" class="retrieval-link">
                View Original
            </a>
        `;
        container.appendChild(item);
    });

    showRetrievalResults();
}

/**
 * Display prediction result
 * @param {Object} data - Prediction response data
 */
function displayResult(data) {
    document.getElementById('resultLabel').textContent = data.predicted_label;
    document.getElementById('resultConfidence').textContent = data.confidence + '%';

    // Display probability distribution
    const probContainer = document.getElementById('probabilities');
    probContainer.innerHTML = '';

    const sortedProbs = Object.entries(data.probabilities)
        .sort((a, b) => b[1] - a[1]);

    sortedProbs.forEach(([label, prob]) => {
        const item = document.createElement('div');
        item.className = 'prob-item';
        item.innerHTML = `
            <div class="prob-label">
                <span class="prob-label-name">${escapeHtml(label)}</span>
                <span class="prob-label-value">${prob.toFixed(2)}%</span>
            </div>
            <div class="prob-bar-container">
                <div class="prob-bar" style="width: ${prob}%"></div>
            </div>
        `;
        probContainer.appendChild(item);
    });

    showResult();
}

/**
 * Set example text in the input field
 * @param {string} text - Example text to set
 */
function setExample(text) {
    document.getElementById('textInput').value = text;
}

/**
 * Clear all inputs and results
 */
function clearAll() {
    document.getElementById('textInput').value = '';
    hideResult();
    hideError();
    hideRetrievalResults();
    hideRetrievalWarning();
}

// ==================== UI Helper Functions ====================

function showLoading(message = 'Analyzing, please wait...') {
    document.getElementById('loadingText').textContent = message;
    document.getElementById('loading').classList.add('active');
    document.querySelector('.btn-predict').disabled = true;
}

function hideLoading() {
    document.getElementById('loading').classList.remove('active');
    document.querySelector('.btn-predict').disabled = false;
}

function showResult() {
    document.getElementById('result').classList.add('active');
}

function hideResult() {
    document.getElementById('result').classList.remove('active');
}

function showRetrievalResults() {
    document.getElementById('retrievalResults').classList.add('active');
}

function hideRetrievalResults() {
    document.getElementById('retrievalResults').classList.remove('active');
}

function showRetrievalWarning(message, isError = false) {
    const warningEl = document.getElementById('retrievalWarning');
    const messageEl = document.getElementById('warningMessage');
    const iconEl = warningEl.querySelector('.warning-icon');
    
    messageEl.textContent = message;
    iconEl.textContent = isError ? '' : '';
    warningEl.classList.toggle('error', isError);
    warningEl.classList.add('active');
}

function hideRetrievalWarning() {
    document.getElementById('retrievalWarning').classList.remove('active');
}

function showError(message) {
    const errorEl = document.getElementById('error');
    errorEl.innerHTML = message; // Use innerHTML to support emoji icons
    errorEl.classList.add('active');
}

function hideError() {
    document.getElementById('error').classList.remove('active');
}

/**
 * Escape HTML to prevent XSS attacks
 * @param {string} text - Text to escape
 * @returns {string} Escaped HTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ==================== Event Listeners ====================

// Enter key submission
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('textInput').addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey) {
            e.preventDefault();
            predict();
        }
    });
});
