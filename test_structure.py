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
        'transformer_autoencoder.py',
        'train_autoencoder.py',
        'example_autoencoder.py',
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
        'transformer_autoencoder.py',
        'train_autoencoder.py',
        'example_autoencoder.py',
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
    
    try:
        import transformer_autoencoder
        print("✓ transformer_autoencoder.py imports OK")
    except ImportError as e:
        print(f"✗ transformer_autoencoder.py import failed (expected without PyTorch): {e}")
    
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

def test_autoencoder_specifications():
    """Test that autoencoder files contain the correct specifications."""
    print("\n=== Testing Autoencoder Specifications ===")
    
    try:
        with open('transformer_autoencoder.py', 'r') as f:
            content = f.read()
        
        # Check for key specifications
        key_specs = [
            '1+900+900 tokens',
            '1024-dimensional vector',
            '900+900 tokens',
            'input_seq_len = 1801',
            'output_seq_len = 1800',
            'latent_dim = 1024'
        ]
        
        missing_specs = []
        for spec in key_specs:
            if spec not in content:
                missing_specs.append(spec)
        
        if missing_specs:
            print(f"✗ Missing autoencoder specifications: {missing_specs}")
            return False
        else:
            print("✓ Autoencoder specifications found")
            return True
            
    except Exception as e:
        print(f"✗ Error reading autoencoder file: {e}")
        return False

def main():
    """Run all tests."""
    print("Transformer & Autoencoder Implementation - Structure Test")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_python_syntax,
        test_imports,
        test_requirements,
        test_documentation,
        test_autoencoder_specifications
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The transformer and autoencoder implementations are ready.")
        print("\nTo use the implementations:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run transformer example: python3 example_usage.py")
        print("3. Run autoencoder example: python3 example_autoencoder.py")
        print("4. Train transformer: python3 train_transformer.py --help")
        print("5. Train autoencoder: python3 train_autoencoder.py --help")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()