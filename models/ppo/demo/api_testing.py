#!/usr/bin/env python3
"""
API Testing Script for PPO Prompt Optimizer
Tests all endpoints and provides comprehensive feedback
"""

import requests
import json
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_PROMPTS = [
    "What is machine learning?",
    "Explain quantum computing",
    "How does photosynthesis work?",
    "Tell me about climate change",
    "What is artificial intelligence?"
]

def test_health_endpoint():
    """Test the health endpoint"""
    print("🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info_endpoint():
    """Test the model info endpoint"""
    print("\n🔍 Testing Model Info Endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/model_info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved: {data}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info error: {e}")
        return False

def test_optimize_prompt_endpoint(prompt):
    """Test the optimize prompt endpoint"""
    print(f"\n🔍 Testing Optimize Prompt: '{prompt[:50]}...'")
    try:
        response = requests.post(
            f"{API_BASE_URL}/optimize_prompt",
            json={"prompt": prompt},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Optimization successful!")
                print(f"   Original: {data.get('original_prompt', 'N/A')}")
                print(f"   Optimized: {data.get('optimized_prompt', 'N/A')}")
                print(f"   Response: {data.get('llm_response', 'N/A')[:100]}...")
                
                # Show metrics if available
                if 'metrics' in data:
                    metrics = data['metrics']
                    print(f"   Metrics: Clarity={metrics.get('clarity_score', 0):.3f}, "
                          f"Relevance={metrics.get('relevance_score', 0):.3f}, "
                          f"Hallucination={metrics.get('hallucination_penalty', 0):.3f}")
                return True
            else:
                print(f"❌ Optimization failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return False

def test_batch_optimization():
    """Test batch optimization with multiple prompts"""
    print(f"\n🔍 Testing Batch Optimization ({len(TEST_PROMPTS)} prompts)...")
    
    results = []
    start_time = time.time()
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n--- Test {i}/{len(TEST_PROMPTS)} ---")
        success = test_optimize_prompt_endpoint(prompt)
        results.append(success)
        
        # Small delay between requests
        if i < len(TEST_PROMPTS):
            time.sleep(1)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n📊 Batch Test Results:")
    print(f"   Total tests: {len(TEST_PROMPTS)}")
    print(f"   Successful: {sum(results)}")
    print(f"   Failed: {len(TEST_PROMPTS) - sum(results)}")
    print(f"   Success rate: {sum(results)/len(TEST_PROMPTS)*100:.1f}%")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   Average time per request: {total_time/len(TEST_PROMPTS):.2f} seconds")
    
    return all(results)

def test_error_handling():
    """Test error handling with invalid requests"""
    print(f"\n🔍 Testing Error Handling...")
    
    # Test 1: Empty prompt
    print("   Testing empty prompt...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/optimize_prompt",
            json={"prompt": ""},
            timeout=10
        )
        print(f"   Empty prompt response: {response.status_code}")
    except Exception as e:
        print(f"   Empty prompt error: {e}")
    
    # Test 2: Missing prompt field
    print("   Testing missing prompt field...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/optimize_prompt",
            json={},
            timeout=10
        )
        print(f"   Missing prompt response: {response.status_code}")
    except Exception as e:
        print(f"   Missing prompt error: {e}")
    
    # Test 3: Invalid JSON
    print("   Testing invalid JSON...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/optimize_prompt",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"   Invalid JSON response: {response.status_code}")
    except Exception as e:
        print(f"   Invalid JSON error: {e}")

def generate_test_report():
    """Generate a comprehensive test report"""
    print("\n" + "="*60)
    print("🚀 PPO PROMPT OPTIMIZER - API TESTING REPORT")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API Base URL: {API_BASE_URL}")
    print("="*60)
    
    # Run all tests
    health_ok = test_health_endpoint()
    model_info_ok = test_model_info_endpoint()
    batch_ok = test_batch_optimization()
    test_error_handling()
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    print(f"Health Endpoint: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Model Info: {'✅ PASS' if model_info_ok else '❌ FAIL'}")
    print(f"Batch Optimization: {'✅ PASS' if batch_ok else '❌ FAIL'}")
    
    if health_ok and model_info_ok and batch_ok:
        print("\n🎉 ALL TESTS PASSED! Your PPO model is working correctly.")
        print("✅ Ready for integration with Streamlit, webapps, and other interfaces.")
    else:
        print("\n⚠️ Some tests failed. Check the API server and model files.")
        print("Make sure to run: docker-compose up --build")
    
    print("="*60)

def main():
    """Main testing function"""
    print("🤖 PPO Prompt Optimizer - API Testing")
    print("Make sure the API server is running: docker-compose up --build")
    print()
    
    try:
        generate_test_report()
    except KeyboardInterrupt:
        print("\n\n⏹️ Testing interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Testing failed with error: {e}")

if __name__ == "__main__":
    main() 