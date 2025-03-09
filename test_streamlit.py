"""
Test script to verify that Streamlit works correctly.
"""
import os
import sys
import streamlit as st

# Add the src directory to the path
sys.path.append(os.path.abspath('src'))

def main():
    """Simple Streamlit app to test that everything works."""
    st.title("Streamlit Test")
    st.write("If you can see this, Streamlit is working correctly!")
    
    st.subheader("Testing Transformers Import")
    try:
        from transformers import AutoModel, AutoTokenizer
        st.success("✅ transformers imported successfully")
        
        import tokenizers
        st.success(f"✅ tokenizers imported successfully (version: {tokenizers.__version__})")
        
        st.info("The environment is correctly set up. You can now run the full app.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.warning("Please run './fix_and_run.sh' to fix the environment.")

if __name__ == "__main__":
    main() 