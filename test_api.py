"""
Test script for the Car Damage Detection API
"""

import requests
from pathlib import Path
import sys


def test_health_endpoint(base_url: str):
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def test_inference(base_url: str, image_path: str, confidence: float = 0.25):
    """Test inference endpoint"""
    print(f"\n🔍 Testing inference with image: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            params = {"confidence": confidence}
            
            response = requests.post(
                f"{base_url}/infer",
                files=files,
                params=params
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Inference successful!")
            print(f"   Total Cost: ${result['total_cost']:.2f}")
            print(f"   Damages Found: {len(result['damages'])}")
            
            for i, damage in enumerate(result['damages'], 1):
                print(f"\n   Damage {i}:")
                print(f"      Part: {damage['part']}")
                print(f"      Severity: {damage['severity']}")
                print(f"      Cost: ${damage['cost']:.2f}")
                print(f"      Confidence: {damage['confidence']:.2f}")
            
            return True
        else:
            print(f"❌ Inference failed: {response.status_code}")
            print(f"   Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main test function"""
    # Configuration
    base_url = "http://localhost:8001"
    
    print("=" * 70)
    print("🧪 Car Damage Detection API Test Suite")
    print("=" * 70)
    print(f"📍 Base URL: {base_url}")
    print("=" * 70)
    
    # Test 1: Health check
    if not test_health_endpoint(base_url):
        print("\n❌ Server is not running or not healthy")
        print("💡 Start the server with: python inference_api.py")
        sys.exit(1)
    
    # Test 2: Inference (if image path provided)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
        test_inference(base_url, image_path, confidence)
    else:
        print("\n💡 To test inference, run:")
        print("   python test_api.py <path_to_image> [confidence]")
        print("   Example: python test_api.py test.jpg 0.3")
    
    print("\n" + "=" * 70)
    print("✅ Tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

