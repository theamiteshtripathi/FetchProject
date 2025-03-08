"""
FastAPI application for serving the multi-task model.
"""
import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sys
import os
# Add the parent directory to the path so we can import the models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.multi_task_model import MultiTaskModel


# Define request and response models
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    sentences: List[str]
    return_probabilities: bool = False


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    task_a_predictions: List[Any]  # Can be list of ints or list of lists (probabilities)
    task_b_predictions: List[List[Any]]  # Can be list of lists of ints or list of lists of lists (probabilities)


# Initialize FastAPI app
app = FastAPI(
    title="Multi-Task Sentence Transformer API",
    description="API for sentence embedding and multi-task predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Global variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task_a_labels = ["Class 0", "Class 1", "Class 2"]  # Example class names
task_b_labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]  # Example NER tags


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model
    
    try:
        # Check if a saved model exists
        model_path = os.environ.get("MODEL_PATH", "./outputs/best_model.pt")
        
        if os.path.exists(model_path):
            # Load saved model
            checkpoint = torch.load(model_path, map_location=device)
            args = checkpoint.get('args', {})
            
            # Create model with the same configuration
            model = MultiTaskModel(
                encoder_model_name=args.get('model_name', 'all-MiniLM-L6-v2'),
                pooling_strategy=args.get('pooling', 'mean'),
                task_a_num_classes=args.get('task_a_classes', 3),
                task_b_num_labels=args.get('task_b_labels', 5),
                max_length=args.get('max_length', 128)
            )
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If no saved model, initialize a new one
            model = MultiTaskModel()
        
        # Move model to device and set to evaluation mode
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Multi-Task Sentence Transformer API"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/encode", response_model=Dict[str, List[List[float]]])
async def encode_sentences(request: PredictionRequest):
    """
    Encode sentences to embeddings.
    
    Args:
        request: PredictionRequest with sentences
        
    Returns:
        Dictionary with sentence embeddings
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Encode sentences
        with torch.no_grad():
            embeddings = model.encoder.encode(request.sentences)
            
        # Convert to Python list
        embeddings_list = embeddings.cpu().numpy().tolist()
        
        return {"embeddings": embeddings_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding sentences: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions for both tasks on the given sentences.
    
    Args:
        request: PredictionRequest with sentences
        
    Returns:
        PredictionResponse with predictions for both tasks
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get predictions
        predictions = model.predict(
            sentences=request.sentences,
            return_probabilities=request.return_probabilities
        )
        
        # Convert numpy arrays to Python lists
        task_a_preds = predictions['task_a'].tolist()
        task_b_preds = predictions['task_b'].tolist()
        
        # Map numeric predictions to labels if not returning probabilities
        if not request.return_probabilities:
            # For Task A (sentence classification)
            task_a_preds = [task_a_labels[pred] for pred in task_a_preds]
            
            # For Task B (token classification)
            # We need to map each token's prediction to its label
            task_b_preds_mapped = []
            for sentence_preds in task_b_preds:
                # Only map up to the actual length of the sentence
                sentence_labels = [task_b_labels[pred] for pred in sentence_preds[:len(sentence_preds)]]
                task_b_preds_mapped.append(sentence_labels)
            task_b_preds = task_b_preds_mapped
        
        return PredictionResponse(
            task_a_predictions=task_a_preds,
            task_b_predictions=task_b_preds
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 