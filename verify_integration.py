#!/usr/bin/env python3
"""
Frontend-ML Integration Verification Script

Run this to verify everything is set up correctly
"""

import os
import sys
from pathlib import Path

def check_file_exists(path, description):
    """Check if file exists"""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def check_directory_exists(path, description):
    """Check if directory exists"""
    exists = os.path.isdir(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def check_python_imports():
    """Check if required packages are installed"""
    packages = [
        'streamlit',
        'pandas',
        'numpy',
        'sklearn',
        'plotly',
    ]
    
    print("\n📦 Checking Python packages...")
    all_installed = True
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            all_installed = False
    
    return all_installed

def check_structure(root):
    """Verify project structure"""
    print("\n📁 Checking project structure...")
    
    checks = [
        # Frontend
        (os.path.join(root, "frontend"), "frontend/", "dir"),
        (os.path.join(root, "frontend/uploads"), "frontend/uploads/", "dir"),
        (os.path.join(root, "frontend/uploads/upload.py"), "frontend/uploads/upload.py", "file"),
        (os.path.join(root, "frontend/pages"), "frontend/pages/", "dir"),
        (os.path.join(root, "frontend/pages/1_training.py"), "frontend/pages/1_training.py", "file"),
        (os.path.join(root, "frontend/pages/2_prediction.py"), "frontend/pages/2_prediction.py", "file"),
        (os.path.join(root, "frontend/pages/3_evaluation.py"), "frontend/pages/3_evaluation.py", "file"),
        (os.path.join(root, "frontend/pages/4_explainability.py"), "frontend/pages/4_explainability.py", "file"),
        (os.path.join(root, "frontend/pages/5_reports.py"), "frontend/pages/5_reports.py", "file"),
        (os.path.join(root, "frontend/utils"), "frontend/utils/", "dir"),
        (os.path.join(root, "frontend/utils/ml_helpers.py"), "frontend/utils/ml_helpers.py", "file"),
        
        # Services
        (os.path.join(root, "services"), "services/", "dir"),
        (os.path.join(root, "services/ml_service.py"), "services/ml_service.py", "file"),
        (os.path.join(root, "services/model_selector.py"), "services/model_selector.py", "file"),
        (os.path.join(root, "services/preprocessing.py"), "services/preprocessing.py", "file"),
        (os.path.join(root, "services/prediction.py"), "services/prediction.py", "file"),
        (os.path.join(root, "services/evaluation.py"), "services/evaluation.py", "file"),
        (os.path.join(root, "services/explainability.py"), "services/explainability.py", "file"),
        (os.path.join(root, "services/report_generator.py"), "services/report_generator.py", "file"),
        (os.path.join(root, "services/model_saver.py"), "services/model_saver.py", "file"),
        
        # Documentation
        (os.path.join(root, "QUICK_START.md"), "QUICK_START.md", "file"),
        (os.path.join(root, "INTEGRATION_GUIDE.md"), "INTEGRATION_GUIDE.md", "file"),
        (os.path.join(root, "INTEGRATION_EXAMPLES.md"), "INTEGRATION_EXAMPLES.md", "file"),
        (os.path.join(root, "README_INTEGRATION.md"), "README_INTEGRATION.md", "file"),
        (os.path.join(root, "INTEGRATION_SUMMARY.md"), "INTEGRATION_SUMMARY.md", "file"),
    ]
    
    all_exist = True
    for path, desc, check_type in checks:
        if check_type == "file":
            if not check_file_exists(path, desc):
                all_exist = False
        elif check_type == "dir":
            if not check_directory_exists(path, desc):
                all_exist = False
    
    return all_exist

def main():
    """Run all checks"""
    print("=" * 60)
    print("🔍 Frontend-ML Integration Verification")
    print("=" * 60)
    
    root = os.path.dirname(os.path.abspath(__file__))
    
    # Check structure
    structure_ok = check_structure(root)
    
    # Check Python packages
    packages_ok = check_python_imports()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Summary")
    print("=" * 60)
    
    if structure_ok and packages_ok:
        print("✅ All checks passed!")
        print("\n🚀 You can now run:")
        print("   streamlit run frontend/uploads/upload.py")
        return 0
    else:
        print("❌ Some checks failed!")
        
        if not structure_ok:
            print("\n📁 Missing files/directories - Check structure")
        
        if not packages_ok:
            print("\n📦 Missing packages - Run: pip install -r requirements.txt")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
