#!/usr/bin/env python3
"""
A2C Model Demo Runner
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = ['torch', 'sentence_transformers', 'flask', 'openai', 'dotenv']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

def check_model_files():
    """Check if model files exist"""
    print("\n🔍 Checking model files...")
    
    required_files = [
        '../model.py',
        '../utils.py'
    ]
    
    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing.append(file_path)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        return False
    
    return True

def check_api_keys():
    """Check if API keys are configured"""
    print("\n🔍 Checking API keys...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    keys = ['GROQ_API_KEY']
    missing = []
    
    for key in keys:
        if os.getenv(key):
            print(f"✅ {key}")
        else:
            print(f"❌ {key} - MISSING")
            missing.append(key)
    
    if missing:
        print(f"\n⚠️  Missing API keys: {', '.join(missing)}")
        print("Add them to your .env file")
        return False
    
    return True

def start_webapp():
    """Start the Flask webapp"""
    print("\n🚀 Starting A2C WebApp...")
    
    try:
        # Start the Flask app
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 WebApp stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting WebApp: {e}")
        return False
    
    return True

def test_api():
    """Test the API endpoint"""
    print("\n🔍 Testing API endpoint...")
    
    try:
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Test status endpoint
        response = requests.get("http://localhost:5000/api/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ API Status: {status}")
            return status.get('success', False)
        else:
            print(f"❌ API Status failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API test failed: {str(e)}")
        print("Make sure the webapp is running on port 5000")
        return False

def main():
    """Main function"""
    print("🎯 A2C Model Demo")
    print("=" * 50)
    
    # Run all checks
    checks = [
        ("Dependencies", check_dependencies),
        ("Model Files", check_model_files),
        ("API Keys", check_api_keys)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        if not check_func():
            all_passed = False
            print(f"\n❌ {check_name} check failed!")
            break
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("✅ All checks passed! Starting A2C WebApp...")
        print("\n🌐 WebApp will be available at: http://localhost:5000")
        print("📱 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Start the webapp
        start_webapp()
    else:
        print("❌ Demo setup failed. Please fix the issues above.")
        print("\n💡 Quick fixes:")
        print("1. Install dependencies: pip install torch sentence-transformers flask openai python-dotenv")
        print("2. Create .env file with GROQ_API_KEY")
        print("3. Ensure model files are in the correct locations")

if __name__ == "__main__":
    main() 