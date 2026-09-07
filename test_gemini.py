import os
import sys
from google import genai

# Verify that Windows has the variable set
if "GEMINI_API_KEY" not in os.environ:
    print("\n[ERROR] GEMINI_API_KEY environment variable not found!")
    sys.exit(1)

# Initialize the client
client = genai.Client()

try:
    print("Sending test request to Gemini...")
    # Switched to the active gemini-3.6-flash model
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents='Hello, are you working?',
    )
    print("\nSUCCESS! Gemini responded:")
    print(response.text)
except Exception as e:
    print(f"\nAPI ERROR: {e}")
