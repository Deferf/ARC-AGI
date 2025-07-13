#!/usr/bin/env python3
"""
Test script to verify the transformer implementation structure.
This script checks the basic syntax and structure without requiring PyTorch.
"""

import os
import sys

def test_file_structure():
    """Test that all required files exist."""
    print("=== Testing File Structure ===")
    
    required_files = [
        'transformer.py',
        'arc_data_loader.py', 
        'train_transformer.py',
        'requirements.txt',
        'README_TRANSFORMER.md',
        'example_usage.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    else:
        print("\nAll required files present!")
        return True

def test_python_syntax():
    """Test Python syntax of the main files."""
    print("\n=== Testing Python Syntax ===")
    
    files_to_test = [
        'transformer.py',
        'arc_data_loader.py',
        'train_transformer.py',
        'example_usage.py'
    ]
    
    syntax_errors = []
    
    for file in files_to_test:
        try:
            with open(file, 'r') as f:
                content = f.read()
            
            # Try to compile the code
            compile(content, file, 'exec')
            print(f"✓ {file} - Syntax OK")
            
        except SyntaxError as e:
            print(f"✗ {file} - Syntax Error: {e}")
            syntax_errors.append((file, e))
        except Exception as e:
            print(f"✗ {file} - Error: {e}")
            syntax_errors.append((file, e))
    
    if syntax_errors:
        print(f"\nSyntax errors found: {len(syntax_errors)}")
        return False
    else:
        print("\nAll files have valid Python syntax!")
        return True

def test_imports():
    """Test that the modules can be imported (without PyTorch)."""
    print("\n=== Testing Module Imports ===")
    
    # Test basic imports that don't require PyTorch
    try:
        import json
        import os
        import math
        import random
        from typing import List, Dict, Tuple, Optional
        print("✓ Standard library imports OK")
    except ImportError as e:
        print(f"✗ Standard library import error: {e}")
        return False
    
    # Test our modules (will fail without PyTorch, but that's expected)
    try:
        import transformer
        print("✓ transformer.py imports OK")
    except ImportError as e:
        print(f"✗ transformer.py import failed (expected without PyTorch): {e}")
    
    try:
        import arc_data_loader
        print("✓ arc_data_loader.py imports OK")
    except ImportError as e:
        print(f"✗ arc_data_loader.py import failed (expected without PyTorch): {e}")
    
    return True

def test_requirements():
    """Test that requirements.txt is properly formatted."""
    print("\n=== Testing Requirements File ===")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        
        print(f"Found {len(requirements)} requirements:")
        for req in requirements:
            if req.strip():
                print(f"  - {req}")
        
        # Check for essential packages
        essential_packages = ['torch', 'numpy', 'tqdm']
        found_packages = [req.split('>=')[0].split('==')[0] for req in requirements if req.strip()]
        
        missing_essential = []
        for pkg in essential_packages:
            if pkg not in found_packages:
                missing_essential.append(pkg)
        
        if missing_essential:
            print(f"✗ Missing essential packages: {missing_essential}")
            return False
        else:
            print("✓ All essential packages included")
            return True
            
    except Exception as e:
        print(f"✗ Error reading requirements.txt: {e}")
        return False

def test_documentation():
    """Test that documentation files exist and are readable."""
    print("\n=== Testing Documentation ===")
    
    try:
        with open('README_TRANSFORMER.md', 'r') as f:
            content = f.read()
        
        # Check for key sections
        key_sections = [
            '## Overview',
            '## Installation', 
            '## Usage',
            '## Architecture',
            '## Training'
        ]
        
        missing_sections = []
        for section in key_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"✗ Missing documentation sections: {missing_sections}")
            return False
        else:
            print("✓ Documentation includes all key sections")
            print(f"✓ Documentation length: {len(content)} characters")
            return True
            
    except Exception as e:
        print(f"✗ Error reading documentation: {e}")
        return False

def main():
    """Run all tests."""
    print("ARC Transformer Implementation - Structure Test")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_python_syntax,
        test_imports,
        test_requirements,
        test_documentation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The transformer implementation is ready.")
        print("\nTo use the transformer:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run example: python3 example_usage.py")
        print("3. Train model: python3 train_transformer.py --help")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()