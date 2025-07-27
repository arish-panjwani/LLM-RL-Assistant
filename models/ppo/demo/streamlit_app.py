#!/usr/bin/env python3
"""
Streamlit App for PPO Prompt Optimization Demo
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="PPO Prompt Optimizer Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200, response.json()
    except requests.exceptions.RequestException:
        return False, None

def optimize_prompt(prompt_text):
    """Send prompt to PPO model for optimization"""
    try:
        response = requests.post(
            "http://localhost:8000/optimize_prompt",
            json={"prompt": prompt_text},
            timeout=30
        )
        return response.status_code == 200, response.json()
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 PPO Prompt Optimizer Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # API Status Check
    st.sidebar.subheader("🔍 API Status")
    api_healthy, api_info = check_api_health()
    
    if api_healthy:
        st.sidebar.success("✅ API Server Running")
        if api_info:
            st.sidebar.json(api_info)
    else:
        st.sidebar.error("❌ API Server Not Found")
        st.sidebar.info("Make sure to run: docker-compose up --build")
        st.stop()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Enter Your Prompt")
        
        # Example prompts
        example_prompts = [
            "What is machine learning?",
            "Explain quantum computing",
            "How does photosynthesis work?",
            "Tell me about climate change",
            "What is artificial intelligence?"
        ]
        
        selected_example = st.selectbox(
            "Or choose an example:",
            ["Custom prompt"] + example_prompts
        )
        
        if selected_example == "Custom prompt":
            user_prompt = st.text_area(
                "Enter your prompt:",
                height=100,
                placeholder="Type your question here..."
            )
        else:
            user_prompt = st.text_area(
                "Enter your prompt:",
                value=selected_example,
                height=100
            )
        
        # Optimize button
        if st.button("🚀 Optimize Prompt", type="primary", use_container_width=True):
            if user_prompt.strip():
                with st.spinner("Optimizing prompt with PPO model..."):
                    success, result = optimize_prompt(user_prompt)
                    
                    if success:
                        st.session_state.result = result
                        st.session_state.timestamp = datetime.now()
                    else:
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
            else:
                st.warning("Please enter a prompt first.")
    
    with col2:
        st.subheader("📊 Results")
        
        if 'result' in st.session_state:
            result = st.session_state.result
            
            # Original vs Optimized
            st.markdown("### 📋 Prompt Comparison")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Original Prompt:**")
                st.info(result.get('original_prompt', 'N/A'))
            
            with col_b:
                st.markdown("**Optimized Prompt:**")
                st.success(result.get('optimized_prompt', 'N/A'))
            
            # LLM Response
            st.markdown("### 🤖 LLM Response")
            st.markdown(result.get('llm_response', 'No response available'))
            
            # Metrics
            if 'metrics' in result:
                st.markdown("### 📈 Performance Metrics")
                
                metrics = result['metrics']
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.metric(
                        "Clarity Score",
                        f"{metrics.get('clarity_score', 0):.3f}",
                        help="How clear and specific the optimized prompt is"
                    )
                
                with col_m2:
                    st.metric(
                        "Relevance Score",
                        f"{metrics.get('relevance_score', 0):.3f}",
                        help="How relevant the response is to the original intent"
                    )
                
                with col_m3:
                    st.metric(
                        "Hallucination Penalty",
                        f"{metrics.get('hallucination_penalty', 0):.3f}",
                        help="Penalty for potential misinformation (lower is better)"
                    )
            
            # Model info
            st.markdown("### 🔧 Model Information")
            st.json({
                "model_used": result.get('model_used', 'PPO'),
                "timestamp": st.session_state.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "success": result.get('success', False)
            })
        else:
            st.info("👆 Enter a prompt and click 'Optimize Prompt' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>PPO Prompt Optimizer Demo</strong> | Built for School Assignment</p>
        <p>Reinforcement Learning for Intelligent Conversational Assistant</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 