"""
Test script to verify that the imports work correctly.
"""
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.abspath('src'))

print("Testing imports...")

try:
    from transformers import AutoModel, AutoTokenizer
    print("✅ transformers imported successfully")
    
    import tokenizers
    print(f"✅ tokenizers imported successfully (version: {tokenizers.__version__})")
    
    from src.models.sentence_encoder import SentenceEncoder
    print("✅ SentenceEncoder imported successfully")
    
    from src.models.multi_task_model import MultiTaskModel
    print("✅ MultiTaskModel imported successfully")
    
    print("\nAll imports successful! The environment is correctly set up.")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nPlease run './fix_and_run.sh' to fix the environment.") 