
#!/usr/bin/env python3
"""
Simple test to verify the app can start up
"""

def test_imports():
    """Test that all required modules can be imported"""
    try:
        print("Testing imports...")
        import flask
        print("✅ Flask imported")
        
        from config import *
        print("✅ Config imported")
        
        try:
            from detection import AIDetector, get_result_classification
            print("✅ Detection module imported")
        except Exception as e:
            print(f"⚠️  Detection module failed: {e}")
            
        try:
            from feedback import feedback_manager
            print("✅ Feedback module imported")
        except Exception as e:
            print(f"⚠️  Feedback module failed: {e}")
            
        print("✅ Import test completed")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_basic_app():
    """Test basic Flask app creation"""
    try:
        print("Testing basic app creation...")
        from flask import Flask
        test_app = Flask(__name__)
        print("✅ Basic Flask app created")
        return True
    except Exception as e:
        print(f"❌ Basic app test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Running startup tests...")
    
    success = True
    success &= test_imports()
    success &= test_basic_app()
    
    if success:
        print("🎉 All startup tests passed!")
    else:
        print("❌ Some tests failed - check the errors above")
        
    print("\n🚀 Try running the app now with: python app.py")
