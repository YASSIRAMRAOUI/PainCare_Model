#!/usr/bin/env python3
"""
Simple YAML syntax checker for GitHub Actions workflows
"""

import sys
import yaml
import os

def check_workflow_syntax(file_path):
    """Check if a YAML workflow file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            yaml.safe_load(file)
        print(f"✅ {file_path} - Valid YAML syntax")
        return True
    except yaml.YAMLError as e:
        print(f"❌ {file_path} - YAML syntax error:")
        print(f"   {e}")
        return False
    except FileNotFoundError:
        print(f"❌ {file_path} - File not found")
        return False

def main():
    """Check all workflow files"""
    workflows_dir = ".github/workflows"
    
    if not os.path.exists(workflows_dir):
        print("❌ No .github/workflows directory found")
        return 1
    
    yaml_files = []
    for file in os.listdir(workflows_dir):
        if file.endswith(('.yml', '.yaml')):
            yaml_files.append(os.path.join(workflows_dir, file))
    
    if not yaml_files:
        print("❌ No YAML workflow files found")
        return 1
    
    print("🔍 Checking GitHub Actions workflow files...")
    print("-" * 50)
    
    all_valid = True
    for file_path in yaml_files:
        if not check_workflow_syntax(file_path):
            all_valid = False
    
    print("-" * 50)
    if all_valid:
        print("✅ All workflow files have valid syntax!")
        return 0
    else:
        print("❌ Some workflow files have syntax errors")
        return 1

if __name__ == "__main__":
    sys.exit(main())
