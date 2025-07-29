#!/usr/bin/env python3
"""
Quick Deployment Script for PPO Model
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class QuickDeploy:
    def __init__(self):
        self.model = None
        self.api_url = "http://localhost:5000"
    
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        print("🔍 Checking dependencies...")
        
        required_packages = ['torch', 'sentence_transformers', 'requests', 'dotenv']
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
    
    def check_model_files(self):
        """Check if model files exist"""
        print("\n🔍 Checking model files...")
        
        required_files = [
            'saved_model/ppo_actor.pth',
            'utils.py',
            'model.py'
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
    
    def check_api_keys(self):
        """Check if API keys are configured"""
        print("\n🔍 Checking API keys...")
        
        keys = ['GROQ_API_KEY', 'WOLFRAM_APP_ID', 'GOOGLE_API_KEY']
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
    
    def test_local_model(self):
        """Test the model locally"""
        print("\n🔍 Testing local model...")
        
        try:
            # Import model components
            sys.path.append('.')
            from utils import PromptEnvironment
            from model import PPOAgent
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Setup model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
            
            state_dim = encoder.get_sentence_embedding_dimension()
            agent = PPOAgent(state_dim=state_dim, action_dim=state_dim, device=device)
            
            # Load trained model
            if os.path.exists('saved_model/ppo_actor.pth'):
                agent.load('saved_model/ppo_actor.pth')
                print("✅ Model loaded successfully")
            else:
                print("⚠️  No trained model found")
            
            # Test optimization
            env = PromptEnvironment(encoder)
            test_prompt = "Hello world"
            env.original_prompt = test_prompt
            
            state = env.encode(test_prompt).unsqueeze(0).to(device)
            action, _, _ = agent.select_action(state)
            action = torch.tensor(action, dtype=torch.float32).to(device)
            optimized = env.decode(action.squeeze())
            
            print("✅ Local model test passed")
            return True
            
        except Exception as e:
            print(f"❌ Local model test failed: {str(e)}")
            return False
    
    def test_api_endpoint(self):
        """Test the API endpoint"""
        print("\n🔍 Testing API endpoint...")
        
        try:
            # Test status endpoint
            response = requests.get(f"{self.api_url}/api/status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                print(f"✅ API Status: {status}")
                return status.get('model_loaded', False)
            else:
                print(f"❌ API Status failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ API test failed: {str(e)}")
            print("Make sure the webapp is running on port 5000")
            return False
    
    def optimize_prompt(self, prompt):
        """Optimize a prompt via API"""
        try:
            response = requests.post(
                f"{self.api_url}/api/optimize",
                json={'prompt': prompt},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            return {'success': False, 'error': str(e)}
    
    def run_demo(self):
        """Run an interactive demo"""
        print("\n🎯 Running Interactive Demo")
        print("=" * 40)
        
        while True:
            prompt = input("\nEnter a prompt to optimize (or 'quit' to exit): ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                print("Please enter a valid prompt")
                continue
            
            print(f"\n🔧 Optimizing: '{prompt}'")
            result = self.optimize_prompt(prompt)
            
            if result.get('success'):
                print(f"\n✅ Original: {result['original']}")
                print(f"🚀 Optimized: {result['optimized']}")
                print(f"🤖 Response: {result['llm_response'][:200]}...")
            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
    
    def deploy(self):
        """Main deployment function"""
        print("🚀 PPO Model Quick Deploy")
        print("=" * 50)
        
        # Run all checks
        checks = [
            ("Dependencies", self.check_dependencies),
            ("Model Files", self.check_model_files),
            ("API Keys", self.check_api_keys),
            ("Local Model", self.test_local_model),
            ("API Endpoint", self.test_api_endpoint)
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            if not check_func():
                all_passed = False
                print(f"\n❌ {check_name} check failed!")
                break
        
        print("\n" + "=" * 50)
        
        if all_passed:
            print("✅ All checks passed! Model is ready for deployment.")
            print("\n🎯 Deployment Options:")
            print("1. Run interactive demo")
            print("2. Test single prompt")
            print("3. Exit")
            
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == "1":
                self.run_demo()
            elif choice == "2":
                prompt = input("Enter test prompt: ").strip()
                if prompt:
                    result = self.optimize_prompt(prompt)
                    print(f"\nResult: {json.dumps(result, indent=2)}")
            else:
                print("👋 Goodbye!")
        else:
            print("❌ Deployment failed. Please fix the issues above.")
            print("\n💡 Quick fixes:")
            print("1. Install dependencies: pip install torch sentence-transformers python-dotenv requests")
            print("2. Create .env file with API keys")
            print("3. Ensure model files are in the correct locations")
            print("4. Start the webapp: python demo/run_demo.py")

if __name__ == "__main__":
    deployer = QuickDeploy()
    deployer.deploy() 