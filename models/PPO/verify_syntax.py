#!/usr/bin/env python3
"""
Syntax verification script for PPO implementation
"""

import ast
import os

def check_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        print(f"✅ {file_path} - Syntax OK")
        return True
    except SyntaxError as e:
        print(f"❌ {file_path} - Syntax Error: {e}")
        return False
    except Exception as e:
        print(f"❌ {file_path} - Error: {e}")
        return False

def main():
    """Check syntax of all Python files"""
    python_files = [
        'model.py',
        'main.py', 
        'utils.py',
        'inference_test.py',
        'test_ppo.py',
        'test_env.py'
    ]
    
    print("🔍 Checking PPO implementation syntax...")
    print("=" * 50)
    
    all_good = True
    for file_path in python_files:
        if os.path.exists(file_path):
            if not check_syntax(file_path):
                all_good = False
        else:
            print(f"⚠️ {file_path} - File not found")
            all_good = False
    
    print("=" * 50)
    if all_good:
        print("🎉 All files have valid syntax!")
    else:
        print("❌ Some files have syntax errors")
    
    return all_good

if __name__ == "__main__":
    main() 