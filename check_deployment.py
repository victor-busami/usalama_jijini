#!/usr/bin/env python3
"""
Deployment verification script for Usalama Jijini
This script checks if all required files are present for successful deployment
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MISSING!")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists and print status"""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ {description}: {dirpath} - MISSING!")
        return False

def main():
    print("🔍 Checking deployment readiness for Usalama Jijini...")
    print("=" * 50)
    
    all_good = True
    
    # Core application files
    all_good &= check_file_exists("app.py", "Main application")
    all_good &= check_file_exists("requirements.txt", "Dependencies")
    all_good &= check_file_exists("render.yaml", "Render config")
    
    # Assets and data
    all_good &= check_directory_exists("assets", "Assets directory")
    all_good &= check_file_exists("assets/hero_safety_image.png", "Hero image")
    all_good &= check_file_exists("yolov8n.pt", "YOLO model")
    all_good &= check_file_exists("reports.db", "Reports database")
    
    # Upload directory
    all_good &= check_directory_exists("uploads", "Uploads directory")
    
    # Git and deployment files
    all_good &= check_file_exists(".gitignore", "Git ignore file")
    all_good &= check_file_exists(".slugignore", "Slug ignore file")
    
    print("=" * 50)
    if all_good:
        print("🎉 All files are present! Ready for deployment.")
        print("\n📋 Next steps:")
        print("1. Commit all changes to Git")
        print("2. Push to your repository")
        print("3. Deploy on Render")
        print("4. The hero image should now be visible!")
    else:
        print("⚠️  Some files are missing. Please fix the issues above before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main()
