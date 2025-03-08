"""
Script to run the API server.
"""
import os
import argparse
import uvicorn


def main():
    """Main function to run the API server."""
    parser = argparse.ArgumentParser(description='Run the API server')
    
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to run the server on')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a saved model checkpoint')
    parser.add_argument('--reload', action='store_true',
                        help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    # Set environment variable for model path if specified
    if args.model_path:
        os.environ['MODEL_PATH'] = args.model_path
    
    # Run the server
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main() 