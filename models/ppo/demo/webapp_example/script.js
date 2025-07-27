// JavaScript for PPO Prompt Optimizer Demo

// API configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM elements
const promptInput = document.getElementById('promptInput');
const optimizeBtn = document.getElementById('optimizeBtn');
const resultsSection = document.getElementById('resultsSection');
const loading = document.getElementById('loading');
const error = document.getElementById('error');

// Example prompt functions
function setPrompt(prompt) {
    promptInput.value = prompt;
    promptInput.focus();
}

// Main optimization function
async function optimizePrompt() {
    const prompt = promptInput.value.trim();
    
    if (!prompt) {
        showError('Please enter a prompt first.');
        return;
    }
    
    // Show loading
    showLoading();
    hideError();
    hideResults();
    
    try {
        const response = await fetch(`${API_BASE_URL}/optimize_prompt`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt: prompt })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            displayResults(result);
        } else {
            throw new Error(result.error || 'Unknown error occurred');
        }
        
    } catch (err) {
        showError(`Failed to optimize prompt: ${err.message}`);
        console.error('Error:', err);
    } finally {
        hideLoading();
    }
}

// Display results
function displayResults(result) {
    // Update DOM elements
    document.getElementById('originalPrompt').textContent = result.original_prompt || 'N/A';
    document.getElementById('optimizedPrompt').textContent = result.optimized_prompt || 'N/A';
    document.getElementById('llmResponse').textContent = result.llm_response || 'No response available';
    
    // Update metrics if available
    if (result.metrics) {
        document.getElementById('clarityScore').textContent = 
            (result.metrics.clarity_score || 0).toFixed(3);
        document.getElementById('relevanceScore').textContent = 
            (result.metrics.relevance_score || 0).toFixed(3);
        document.getElementById('hallucinationPenalty').textContent = 
            (result.metrics.hallucination_penalty || 0).toFixed(3);
    }
    
    // Show results with animation
    resultsSection.style.display = 'block';
    resultsSection.classList.add('success-animation');
    
    // Remove animation class after animation completes
    setTimeout(() => {
        resultsSection.classList.remove('success-animation');
    }, 500);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// UI helper functions
function showLoading() {
    loading.style.display = 'block';
    optimizeBtn.disabled = true;
    optimizeBtn.textContent = '⏳ Processing...';
}

function hideLoading() {
    loading.style.display = 'none';
    optimizeBtn.disabled = false;
    optimizeBtn.textContent = '🚀 Optimize Prompt';
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    error.style.display = 'block';
}

function hideError() {
    error.style.display = 'none';
}

function hideResults() {
    resultsSection.style.display = 'none';
}

// Health check on page load
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('✅ API server is running');
        } else {
            console.warn('⚠️ API server might not be running');
        }
    } catch (err) {
        console.warn('⚠️ Cannot connect to API server. Make sure to run: docker-compose up --build');
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Check API health on load
    checkApiHealth();
    
    // Add enter key support for textarea
    promptInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            e.preventDefault();
            optimizePrompt();
        }
    });
    
    // Add click outside to clear error
    document.addEventListener('click', function(e) {
        if (!error.contains(e.target)) {
            hideError();
        }
    });
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+Enter to optimize
    if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        optimizePrompt();
    }
    
    // Escape to clear input
    if (e.key === 'Escape') {
        promptInput.value = '';
        hideError();
        hideResults();
    }
});

// Utility functions for external use
window.PPOOptimizer = {
    optimizePrompt,
    setPrompt,
    checkApiHealth
}; 