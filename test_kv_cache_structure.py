#!/usr/bin/env python3
"""
Simple test script to verify KV cache code structure.
This script checks the syntax and basic structure without requiring PyTorch.
"""

import ast
import sys
from pathlib import Path

def check_python_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def check_file_structure():
    """Check the structure of the KV cache implementation files."""
    files_to_check = [
        'kv_cache.py',
        'transformer_with_kv_cache.py',
        'kv_cache_demo.py'
    ]
    
    print("Checking KV Cache Implementation Structure")
    print("=" * 50)
    
    all_good = True
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"✓ {file_path} exists")
            if check_python_syntax(file_path):
                print(f"✓ {file_path} has valid Python syntax")
            else:
                print(f"✗ {file_path} has syntax errors")
                all_good = False
        else:
            print(f"✗ {file_path} not found")
            all_good = False
    
    return all_good

def check_class_definitions():
    """Check for expected class definitions in the files."""
    print("\nChecking Class Definitions")
    print("=" * 30)
    
    expected_classes = {
        'kv_cache.py': ['KVCache', 'CachedMultiHeadAttention', 'CachedTransformerDecoderLayer', 
                       'CachedTransformerDecoder', 'OptimizedTransformer'],
        'transformer_with_kv_cache.py': ['EnhancedMultiHeadAttention', 'EnhancedTransformerDecoderLayer',
                                        'EnhancedTransformerDecoder', 'EnhancedTransformer'],
        'kv_cache_demo.py': ['NaiveTransformer']
    }
    
    all_good = True
    
    for file_path, expected_classes_list in expected_classes.items():
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                found_classes = []
                for class_name in expected_classes_list:
                    if f"class {class_name}" in content:
                        found_classes.append(class_name)
                
                missing_classes = set(expected_classes_list) - set(found_classes)
                if missing_classes:
                    print(f"✗ {file_path} missing classes: {missing_classes}")
                    all_good = False
                else:
                    print(f"✓ {file_path} has all expected classes")
            except Exception as e:
                print(f"✗ Error reading {file_path}: {e}")
                all_good = False
        else:
            print(f"✗ {file_path} not found")
            all_good = False
    
    return all_good

def check_function_definitions():
    """Check for expected function definitions."""
    print("\nChecking Function Definitions")
    print("=" * 30)
    
    expected_functions = {
        'kv_cache.py': ['benchmark_generation', 'create_causal_mask'],
        'kv_cache_demo.py': ['benchmark_comparison', 'plot_benchmark_results', 
                            'demonstrate_memory_usage', 'demonstrate_cache_behavior'],
        'transformer_with_kv_cache.py': ['test_compatibility']
    }
    
    all_good = True
    
    for file_path, expected_functions_list in expected_functions.items():
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                found_functions = []
                for func_name in expected_functions_list:
                    if f"def {func_name}" in content:
                        found_functions.append(func_name)
                
                missing_functions = set(expected_functions_list) - set(found_functions)
                if missing_functions:
                    print(f"✗ {file_path} missing functions: {missing_functions}")
                    all_good = False
                else:
                    print(f"✓ {file_path} has all expected functions")
            except Exception as e:
                print(f"✗ Error reading {file_path}: {e}")
                all_good = False
        else:
            print(f"✗ {file_path} not found")
            all_good = False
    
    return all_good

def check_documentation():
    """Check for documentation files."""
    print("\nChecking Documentation")
    print("=" * 20)
    
    doc_files = ['README_KV_CACHE.md']
    all_good = True
    
    for doc_file in doc_files:
        if Path(doc_file).exists():
            print(f"✓ {doc_file} exists")
        else:
            print(f"✗ {doc_file} not found")
            all_good = False
    
    return all_good

def main():
    """Main test function."""
    print("KV Cache Implementation Structure Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", check_file_structure),
        ("Class Definitions", check_class_definitions),
        ("Function Definitions", check_function_definitions),
        ("Documentation", check_documentation)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if not test_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! KV cache implementation structure is correct.")
        print("\nThe implementation includes:")
        print("- Core KV cache functionality (kv_cache.py)")
        print("- Enhanced transformer with backward compatibility (transformer_with_kv_cache.py)")
        print("- Demonstration and benchmarking tools (kv_cache_demo.py)")
        print("- Comprehensive documentation (README_KV_CACHE.md)")
    else:
        print("✗ Some tests failed. Please check the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    main()