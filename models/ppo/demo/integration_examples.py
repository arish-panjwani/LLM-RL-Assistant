#!/usr/bin/env python3
"""
Integration Examples for PPO Prompt Optimizer
Shows how to integrate the PPO model with different platforms and use cases
"""

import requests
import json
import time
from typing import Dict, Any, Optional

# Configuration
API_BASE_URL = "http://localhost:8000"

class PPOClient:
    """Client class for interacting with PPO Prompt Optimizer API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
    
    def health_check(self) -> bool:
        """Check if the API server is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def optimize_prompt(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Optimize a prompt using the PPO model"""
        try:
            response = requests.post(
                f"{self.base_url}/optimize_prompt",
                json={"prompt": prompt},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: HTTP {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get model information"""
        try:
            response = requests.get(f"{self.base_url}/model_info", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.exceptions.RequestException:
            return None

# Example 1: Simple Python Integration
def simple_integration_example():
    """Simple integration example"""
    print("=== Simple Python Integration ===")
    
    client = PPOClient()
    
    if not client.health_check():
        print("❌ API server not available")
        return
    
    prompt = "What is machine learning?"
    print(f"Original prompt: {prompt}")
    
    result = client.optimize_prompt(prompt)
    if result and result.get('success'):
        print(f"Optimized prompt: {result['optimized_prompt']}")
        print(f"LLM response: {result['llm_response'][:100]}...")
    else:
        print("❌ Optimization failed")

# Example 2: Batch Processing
def batch_processing_example():
    """Batch processing example"""
    print("\n=== Batch Processing Example ===")
    
    client = PPOClient()
    
    prompts = [
        "Explain quantum computing",
        "How does photosynthesis work?",
        "Tell me about climate change",
        "What is artificial intelligence?"
    ]
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\nProcessing {i}/{len(prompts)}: {prompt}")
        result = client.optimize_prompt(prompt)
        
        if result and result.get('success'):
            results.append({
                'original': prompt,
                'optimized': result['optimized_prompt'],
                'response': result['llm_response'][:100] + "..."
            })
            print(f"✅ Success")
        else:
            print(f"❌ Failed")
    
    print(f"\nBatch processing complete: {len(results)}/{len(prompts)} successful")

# Example 3: Streamlit Integration Helper
def streamlit_integration_helper():
    """Helper functions for Streamlit integration"""
    print("\n=== Streamlit Integration Helper ===")
    
    # This would be used in a Streamlit app
    streamlit_code = '''
import streamlit as st
import requests

def optimize_prompt_streamlit(prompt):
    """Optimize prompt for Streamlit app"""
    try:
        response = requests.post(
            "http://localhost:8000/optimize_prompt",
            json={"prompt": prompt},
            timeout=30
        )
        return response.json() if response.status_code == 200 else None
    except:
        return None

# Streamlit app code
st.title("PPO Prompt Optimizer")
user_input = st.text_input("Enter your prompt:")
if st.button("Optimize"):
    if user_input:
        with st.spinner("Optimizing..."):
            result = optimize_prompt_streamlit(user_input)
            if result and result.get('success'):
                st.success("Optimization successful!")
                st.write("Original:", result['original_prompt'])
                st.write("Optimized:", result['optimized_prompt'])
                st.write("Response:", result['llm_response'])
            else:
                st.error("Optimization failed")
'''
    
    print("Streamlit integration code:")
    print(streamlit_code)

# Example 4: WebApp Integration Helper
def webapp_integration_helper():
    """Helper functions for webapp integration"""
    print("\n=== WebApp Integration Helper ===")
    
    # JavaScript code for webapp
    js_code = '''
// JavaScript for webapp integration
async function optimizePrompt(prompt) {
    try {
        const response = await fetch('http://localhost:8000/optimize_prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt })
        });
        
        if (response.ok) {
            return await response.json();
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
}

// Usage in webapp
document.getElementById('optimizeBtn').addEventListener('click', async () => {
    const prompt = document.getElementById('promptInput').value;
    const result = await optimizePrompt(prompt);
    
    if (result && result.success) {
        document.getElementById('optimizedPrompt').textContent = result.optimized_prompt;
        document.getElementById('llmResponse').textContent = result.llm_response;
    } else {
        alert('Optimization failed');
    }
});
'''
    
    print("JavaScript integration code:")
    print(js_code)

# Example 5: Raspberry Pi Integration
def raspberry_pi_integration():
    """Raspberry Pi integration example"""
    print("\n=== Raspberry Pi Integration ===")
    
    pi_code = '''
# Raspberry Pi integration example
import requests
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# PPO API configuration
PPO_API_URL = "http://localhost:8000"

@app.route('/optimize', methods=['POST'])
def optimize_prompt():
    """Optimize prompt for mobile app"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        # Call PPO API
        response = requests.post(
            f"{PPO_API_URL}/optimize_prompt",
            json={"prompt": prompt},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                'success': True,
                'optimized_prompt': result.get('optimized_prompt'),
                'llm_response': result.get('llm_response')
            })
        else:
            return jsonify({'success': False, 'error': 'API error'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
'''
    
    print("Raspberry Pi integration code:")
    print(pi_code)

# Example 6: Performance Monitoring
def performance_monitoring_example():
    """Performance monitoring example"""
    print("\n=== Performance Monitoring ===")
    
    client = PPOClient()
    
    test_prompts = [
        "What is machine learning?",
        "Explain quantum computing",
        "How does photosynthesis work?"
    ]
    
    performance_data = []
    
    for prompt in test_prompts:
        start_time = time.time()
        result = client.optimize_prompt(prompt)
        end_time = time.time()
        
        if result and result.get('success'):
            performance_data.append({
                'prompt': prompt,
                'response_time': end_time - start_time,
                'success': True,
                'metrics': result.get('metrics', {})
            })
        else:
            performance_data.append({
                'prompt': prompt,
                'response_time': end_time - start_time,
                'success': False
            })
    
    # Calculate statistics
    successful_requests = [d for d in performance_data if d['success']]
    avg_response_time = sum(d['response_time'] for d in successful_requests) / len(successful_requests) if successful_requests else 0
    
    print(f"Performance Summary:")
    print(f"  Total requests: {len(performance_data)}")
    print(f"  Successful: {len(successful_requests)}")
    print(f"  Success rate: {len(successful_requests)/len(performance_data)*100:.1f}%")
    print(f"  Average response time: {avg_response_time:.2f} seconds")
    
    if successful_requests:
        avg_clarity = sum(d['metrics'].get('clarity_score', 0) for d in successful_requests) / len(successful_requests)
        avg_relevance = sum(d['metrics'].get('relevance_score', 0) for d in successful_requests) / len(successful_requests)
        print(f"  Average clarity score: {avg_clarity:.3f}")
        print(f"  Average relevance score: {avg_relevance:.3f}")

def main():
    """Run all integration examples"""
    print("🤖 PPO Prompt Optimizer - Integration Examples")
    print("Make sure the API server is running: docker-compose up --build")
    print()
    
    # Check if API is available
    client = PPOClient()
    if not client.health_check():
        print("❌ API server not available. Please start the server first.")
        print("Run: docker-compose up --build")
        return
    
    print("✅ API server is running. Running integration examples...")
    print()
    
    # Run all examples
    simple_integration_example()
    batch_processing_example()
    streamlit_integration_helper()
    webapp_integration_helper()
    raspberry_pi_integration()
    performance_monitoring_example()
    
    print("\n🎉 All integration examples completed!")
    print("Your PPO model is ready for deployment with any of these integrations.")

if __name__ == "__main__":
    main() 