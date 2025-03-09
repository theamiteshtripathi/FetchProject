"""
Script to run the Streamlit frontend.
"""
import os
import subprocess
import argparse


def main():
    """Run the Streamlit frontend."""
    parser = argparse.ArgumentParser(description='Run the Streamlit frontend')
    
    parser.add_argument('--port', type=int, default=8501,
                        help='Port to run the Streamlit app on')
    
    args = parser.parse_args()
    
    # Get the path to the frontend app
    frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'app.py')
    
    # Run the Streamlit app
    cmd = [
        'streamlit', 'run', frontend_path,
        '--server.port', str(args.port)
    ]
    
    print(f"Starting Streamlit frontend on port {args.port}...")
    print(f"Open your browser at http://localhost:{args.port}")
    
    subprocess.run(cmd)


if __name__ == "__main__":
    main() 