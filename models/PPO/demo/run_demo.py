#!/usr/bin/env python3
"""
Demo Runner with Comprehensive Pre-flight Checks
"""

import sys
import os
import subprocess
import importlib

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = ['flask', 'torch', 'sentence_transformers']
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies available")
    return True

def check_model_files():
    """Check if required model files exist"""
    print("\n🔍 Checking model files...")
    
    required_files = [
        "../utils.py",
        "../model.py",
        "../saved_model/ppo_actor.pth"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All model files available")
    return True

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("\n🔍 Testing model loading...")
    
    try:
        # Test imports
        from utils import PromptEnvironment
        from model import PPOAgent
        from sentence_transformers import SentenceTransformer
        import torch
        print("✅ Imports successful")
        
        # Test device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Device: {device}")
        
        # Test encoder
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Sentence transformer loaded")
        
        # Test agent creation
        state_dim = encoder.get_sentence_embedding_dimension()
        action_dim = state_dim
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)
        print("✅ PPO agent created")
        
        # Test model loading
        model_path = "../saved_model/ppo_actor.pth"
        if os.path.exists(model_path):
            agent.load(model_path)
            print("✅ Pre-trained model loaded")
        else:
            print("⚠️  No pre-trained model found")
        
        # Test environment
        env = PromptEnvironment(encoder)
        print("✅ Environment created")
        
        # Test full optimization pipeline
        test_prompt = "Hello world"
        env.original_prompt = test_prompt
        state = env.encode(test_prompt).unsqueeze(0).to(device)
        action, _, _ = agent.select_action(state)
        action = torch.tensor(action, dtype=torch.float32).to(device)
        optimized_prompt = env.decode(action.squeeze())
        print("✅ Full optimization pipeline test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_env_file():
    """Check if .env file exists"""
    print("\n🔍 Checking .env file...")
    
    env_path = "../.env"
    if os.path.exists(env_path):
        print("✅ .env file found")
        return True
    else:
        print("❌ .env file missing")
        print("Please create a .env file with your API keys:")
        print("GROQ_API_KEY=your_groq_key")
        print("WOLFRAM_APP_ID=your_wolfram_id")
        print("GOOGLE_API_KEY=your_google_key")
        return False

def test_api_keys():
    """Test if API keys are properly loaded"""
    print("\n🔍 Testing API keys...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv("../.env")
        
        import os
        groq_key = os.getenv("GROQ_API_KEY")
        wolfram_id = os.getenv("WOLFRAM_APP_ID")
        google_key = os.getenv("GOOGLE_API_KEY")
        
        if groq_key:
            print("✅ GROQ_API_KEY loaded")
        else:
            print("❌ GROQ_API_KEY missing")
        
        if wolfram_id:
            print("✅ WOLFRAM_APP_ID loaded")
        else:
            print("❌ WOLFRAM_APP_ID missing")
        
        if google_key:
            print("✅ GOOGLE_API_KEY loaded")
        else:
            print("❌ GOOGLE_API_KEY missing")
        
        if groq_key and wolfram_id and google_key:
            print("✅ All API keys loaded")
            return True
        else:
            print("⚠️  Some API keys are missing")
            return False
            
    except Exception as e:
        print(f"❌ API key test failed: {str(e)}")
        return False

def main():
    """Main function to run all checks and start the demo"""
    print("🚀 PPO Demo - Pre-flight Checks")
    print("=" * 50)
    
    # Run all checks
    checks = [
        ("Dependencies", check_dependencies),
        ("Model Files", check_model_files),
        ("Model Loading", test_model_loading),
        ("Environment File", check_env_file),
        ("API Keys", test_api_keys)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        if not check_func():
            all_passed = False
            print(f"\n❌ {check_name} check failed!")
            break
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("✅ All checks passed! Starting webapp...")
        print("\n🌐 Starting Flask webapp...")
        print("📱 Open your browser to: http://localhost:5000")
        print("🔧 API endpoints:")
        print("   - Status: http://localhost:5000/api/status")
        print("   - Test: http://localhost:5000/api/test")
        print("   - Debug: http://localhost:5000/api/debug")
        print("\nPress Ctrl+C to stop the server")
        
        # Start the Flask app
        try:
            # Use the correct Python executable
            python_exe = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ppo_env", "Scripts", "python.exe")
            if os.path.exists(python_exe):
                subprocess.run([python_exe, "app.py"], cwd=os.path.dirname(__file__))
            else:
                print("⚠️  Virtual environment not found, using system Python")
                subprocess.run([sys.executable, "app.py"], cwd=os.path.dirname(__file__))
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
        except Exception as e:
            print(f"\n❌ Error starting demo: {str(e)}")
    else:
        print("❌ Some checks failed. Please fix the issues above before running the demo.")
        print("\n💡 Quick fixes:")
        print("1. Install missing packages: pip install flask torch sentence_transformers python-dotenv requests")
        print("2. Create .env file with your API keys")
        print("3. Ensure model files are in the correct locations")

if __name__ == "__main__":
    main() 