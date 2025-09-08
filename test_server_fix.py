#!/usr/bin/env python3
"""
Test script to verify AssemblyAI integration works with the server fix
"""
import json
import sys
import subprocess
import time
import requests
import os

def test_server_startup():
    """Test that the server can start with AssemblyAI integration"""
    print("🧪 Testing server startup with AssemblyAI integration...")
    
    # Test the server import sequence
    try:
        # This simulates what happens when the server starts
        sys.path.insert(0, 'local_setup')
        
        # Import with the same sequence as the fixed server
        import importlib
        if 'bolna.providers' in sys.modules:
            importlib.reload(sys.modules['bolna.providers'])
        if 'bolna.models' in sys.modules:
            importlib.reload(sys.modules['bolna.models'])
        
        from bolna.models import Transcriber
        from bolna.providers import SUPPORTED_TRANSCRIBER_PROVIDERS
        
        print(f"✅ Available providers: {list(SUPPORTED_TRANSCRIBER_PROVIDERS.keys())}")
        
        # Test the specific validation that was failing
        transcriber = Transcriber(
            provider='assembly',
            encoding='linear16', 
            language='en',
            stream=True,
            sampling_rate=16000
        )
        print("✅ Assembly transcriber validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Server startup test failed: {e}")
        return False

def test_agent_config():
    """Test that agent configuration with AssemblyAI validates"""
    print("\n🤖 Testing agent configuration with AssemblyAI...")
    
    try:
        # Load the test agent configuration
        with open('test_assembly_agent.json', 'r') as f:
            agent_data = json.load(f)
        
        # Import and test the agent model
        from bolna.models import AgentModel
        agent = AgentModel(**agent_data['agent_config'])
        
        transcriber_config = agent.tasks[0].tools_config.transcriber
        print(f"✅ Agent transcriber provider: {transcriber_config.provider}")
        print(f"✅ Agent transcriber language: {transcriber_config.language}")
        print(f"✅ Agent transcriber encoding: {transcriber_config.encoding}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing AssemblyAI integration fix...\n")
    
    
    test1_passed = test_server_startup()
    
    
    test2_passed = test_agent_config()
    
   
    print(f"\n📊 Test Results:")
    print(f"  Server startup: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"  Agent config:   {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"\n📋 Next steps:")
        print(f"1. Kill any running servers: pkill -f uvicorn")
        print(f"2. Start fresh server: uvicorn quickstart_server:app --app-dir local_setup/ --port 5001 --reload")
        print(f"3. Look for log: 'Loaded transcriber providers: ['deepgram', 'whisper', 'azure', 'assembly']'")
        print(f"4. Test agent creation: curl -X POST http://localhost:5001/agent -H 'Content-Type: application/json' -d @test_assembly_agent.json")
        print(f"\n✨ AssemblyAI integration is ready!")
    else:
        print(f"\n❌ Some tests failed. Check the error messages above.")
    
    return test1_passed and test2_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)