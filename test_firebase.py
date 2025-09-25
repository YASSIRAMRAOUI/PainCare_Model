#!/usr/bin/env python3
"""
Firebase Connection Test Script
Tests Firebase connection and displays helpful debug information
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_firebase_credentials():
    """Check Firebase credentials file"""
    print("🔍 Checking Firebase Credentials...")
    
    # Check for service account file
    service_account_paths = [
        "firebase-service-account.json",
        "./firebase-service-account.json",
        os.path.join(os.getcwd(), "firebase-service-account.json")
    ]
    
    found_file = None
    for path in service_account_paths:
        if os.path.exists(path):
            found_file = path
            break
    
    if not found_file:
        print("❌ Firebase service account file not found!")
        print("   Expected locations:")
        for path in service_account_paths:
            print(f"   - {path}")
        print("\n💡 To fix this:")
        print("   1. Go to Firebase Console → Project Settings → Service Accounts")
        print("   2. Click 'Generate new private key'")
        print("   3. Save the file as 'firebase-service-account.json' in the project root")
        return False
    
    print(f"✅ Found service account file: {found_file}")
    
    # Validate JSON structure
    try:
        with open(found_file, 'r') as f:
            creds = json.load(f)
        
        required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
        missing_fields = [field for field in required_fields if field not in creds]
        
        if missing_fields:
            print(f"❌ Missing required fields in credentials: {missing_fields}")
            return False
        
        print(f"✅ Credentials valid for project: {creds.get('project_id')}")
        print(f"   Service account: {creds.get('client_email')}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in credentials file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading credentials file: {e}")
        return False

def check_environment():
    """Check environment variables"""
    print("\n🌍 Checking Environment Variables...")
    
    # Check .env file
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"✅ Found .env file: {env_file}")
        try:
            with open(env_file, 'r') as f:
                env_content = f.read()
            
            # Check for Firebase-related variables
            firebase_vars = ['FIREBASE_SERVICE_ACCOUNT_PATH', 'FIREBASE_DATABASE_URL']
            for var in firebase_vars:
                if var in env_content:
                    # Get the value
                    for line in env_content.split('\n'):
                        if line.startswith(f"{var}="):
                            value = line.split('=', 1)[1].strip()
                            print(f"   {var}={value}")
                else:
                    print(f"   {var}=<not set>")
        except Exception as e:
            print(f"❌ Error reading .env file: {e}")
    else:
        print("⚠️  No .env file found (using default configuration)")
    
    # Check system environment variables
    firebase_env_vars = {
        'FIREBASE_SERVICE_ACCOUNT_PATH': os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH'),
        'FIREBASE_DATABASE_URL': os.getenv('FIREBASE_DATABASE_URL'),
    }
    
    print("\n   System environment variables:")
    for var, value in firebase_env_vars.items():
        if value:
            print(f"   {var}={value}")
        else:
            print(f"   {var}=<not set>")

def test_firebase_connection():
    """Test actual Firebase connection"""
    print("\n🔥 Testing Firebase Connection...")
    
    try:
        from src.services.firebase_service import FirebaseService
        print("✅ Firebase service module imported successfully")
        
        # Initialize service
        firebase_service = FirebaseService()
        
        if firebase_service.db is None:
            print("❌ Firebase database connection failed")
            return False
        
        print("✅ Firebase database connected successfully")
        
        # Test basic query
        try:
            users_collection = firebase_service.db.collection('Users')
            # Try to get collection reference (doesn't read data)
            users_ref = users_collection.limit(1)
            docs = list(users_ref.stream())
            print(f"✅ Successfully accessed Users collection ({len(docs)} documents found)")
        except Exception as e:
            print(f"⚠️  Could not access Users collection: {e}")
            print("   This might be due to Firestore rules or empty collection")
        
        # Test Symptoms collection
        try:
            symptoms_collection = firebase_service.db.collection('Symptoms')
            symptoms_ref = symptoms_collection.limit(1)
            docs = list(symptoms_ref.stream())
            print(f"✅ Successfully accessed Symptoms collection ({len(docs)} documents found)")
        except Exception as e:
            print(f"⚠️  Could not access Symptoms collection: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import Firebase service: {e}")
        print("   Make sure requirements are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Firebase connection test failed: {e}")
        return False

def main():
    print("🚀 PainCare AI - Firebase Connection Test")
    print("=" * 50)
    
    # Check credentials
    creds_ok = check_firebase_credentials()
    
    # Check environment
    check_environment()
    
    if not creds_ok:
        print("\n❌ Cannot proceed with connection test due to credential issues")
        return
    
    # Test connection
    connection_ok = test_firebase_connection()
    
    print("\n" + "=" * 50)
    if connection_ok:
        print("🎉 Firebase connection test PASSED!")
        print("   Your management interface should work with Firebase data.")
    else:
        print("💥 Firebase connection test FAILED!")
        print("   The management interface will use sample data instead.")
    
    print("\n💡 Next steps:")
    if connection_ok:
        print("   - Your Firebase is connected and ready!")
        print("   - Run: python start.py")
        print("   - Open: http://localhost:5000")
    else:
        print("   - Fix the Firebase connection issues above")
        print("   - Run this test again to verify")
        print("   - Check Firebase Console for project settings")

if __name__ == "__main__":
    main()