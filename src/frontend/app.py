"""
Streamlit app for the Multi-Task Sentence Transformer.
"""
import os
import sys
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Add the parent directory to the path so we can import the models
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Import the models
from src.models.sentence_encoder import SentenceEncoder
from src.models.multi_task_model import MultiTaskModel

# Set page config
st.set_page_config(
    page_title="Multi-Task Sentence Transformer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BBE;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #306998;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .task-header {
        font-size: 1.2rem;
        color: #FFD43B;
        background-color: #306998;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .highlight {
        background-color: #FFD43B;
        padding: 0.2rem;
        border-radius: 3px;
    }
    .token-label {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        margin: 0.1rem;
        border-radius: 3px;
    }
    .token-O {
        background-color: #E5E5E5;
    }
    .token-B-PER, .token-I-PER {
        background-color: #FFD43B;
    }
    .token-B-ORG, .token-I-ORG {
        background-color: #4B8BBE;
        color: white;
    }
    .token-B-LOC, .token-I-LOC {
        background-color: #306998;
        color: white;
    }
    .token-B-MISC, .token-I-MISC {
        background-color: #646464;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_name, pooling_strategy):
    """
    Load the model and cache it.
    
    Args:
        model_name: Name of the pre-trained model
        pooling_strategy: Pooling strategy for sentence embeddings
        
    Returns:
        Tuple of (encoder, model)
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load sentence encoder
    encoder = SentenceEncoder(model_name=model_name, pooling_strategy=pooling_strategy)
    encoder = encoder.to(device)
    encoder.eval()
    
    # Load multi-task model
    model = MultiTaskModel(
        encoder_model_name=model_name,
        pooling_strategy=pooling_strategy,
        task_a_num_classes=2,  # SST-2: 2 classes (negative, positive)
        task_b_num_labels=9    # CoNLL-2003: 9 labels (O, B-PER, I-PER, etc.)
    )
    model = model.to(device)
    model.eval()
    
    return encoder, model

def encode_sentences(encoder, sentences):
    """
    Encode sentences and return embeddings.
    
    Args:
        encoder: Sentence encoder
        sentences: List of sentences
        
    Returns:
        Numpy array of embeddings
    """
    with torch.no_grad():
        embeddings = encoder.encode(sentences)
    return embeddings.cpu().numpy()

def get_similarity_matrix(embeddings):
    """
    Compute similarity matrix for embeddings.
    
    Args:
        embeddings: Numpy array of embeddings
        
    Returns:
        Similarity matrix
    """
    return cosine_similarity(embeddings)

def predict_tasks(model, sentences):
    """
    Make predictions for both tasks.
    
    Args:
        model: Multi-task model
        sentences: List of sentences
        
    Returns:
        Dictionary with predictions and probabilities
    """
    # Get predictions (class indices)
    predictions = model.predict(sentences, return_probabilities=False)
    
    # Get probabilities
    probabilities = model.predict(sentences, return_probabilities=True)
    
    return {
        'task_a_preds': predictions['task_a'],
        'task_b_preds': predictions['task_b'],
        'task_a_probs': probabilities['task_a'],
        'task_b_probs': probabilities['task_b']
    }

def main():
    """Main function for the Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">Multi-Task Sentence Transformer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Model Configuration")
    
    model_name = st.sidebar.selectbox(
        "Pre-trained Model",
        ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "bert-base-uncased"],
        index=0
    )
    
    pooling_strategy = st.sidebar.selectbox(
        "Pooling Strategy",
        ["mean", "cls", "max"],
        index=0
    )
    
    # Load model
    with st.spinner("Loading model..."):
        encoder, model = load_model(model_name, pooling_strategy)
    
    # Input section
    st.markdown('<h2 class="sub-header">Enter Sentences</h2>', unsafe_allow_html=True)
    
    # Example sentences
    example_sentences = [
        "I love machine learning and natural language processing.",
        "Deep learning models are revolutionizing NLP applications.",
        "The weather is beautiful today.",
        "It's a sunny day with clear skies.",
        "Python is my favorite programming language.",
        "Apple Inc. is headquartered in Cupertino, California.",
        "Barack Obama was the 44th president of the United States.",
        "I really enjoyed the movie, it was fantastic!",
        "The weather in New York is terrible today."
    ]
    
    # Allow user to select example sentences or enter their own
    use_examples = st.checkbox("Use example sentences", value=True)
    
    if use_examples:
        selected_examples = st.multiselect(
            "Select example sentences",
            example_sentences,
            default=example_sentences[:3]
        )
        sentences = selected_examples
    else:
        user_input = st.text_area("Enter your own sentences (one per line)", height=150)
        sentences = [s.strip() for s in user_input.split('\n') if s.strip()]
    
    # Check if we have sentences to process
    if not sentences:
        st.warning("Please enter at least one sentence.")
        return
    
    # Process button
    if st.button("Process Sentences"):
        # Encode sentences
        with st.spinner("Encoding sentences..."):
            embeddings = encode_sentences(encoder, sentences)
            similarity_matrix = get_similarity_matrix(embeddings)
        
        # Make predictions
        with st.spinner("Making predictions..."):
            results = predict_tasks(model, sentences)
        
        # Display results
        st.markdown('<h2 class="sub-header">Results</h2>', unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Sentence Embeddings", "Task Predictions", "Similarity Matrix"])
        
        with tab1:
            st.markdown('<h3 class="task-header">Sentence Embeddings</h3>', unsafe_allow_html=True)
            
            # Show embedding dimensions
            st.write(f"Embedding shape: {embeddings.shape}")
            
            # Plot first few dimensions of each embedding
            fig, ax = plt.subplots(figsize=(10, 6))
            num_dims = min(10, embeddings.shape[1])
            
            for i, sentence in enumerate(sentences):
                ax.plot(embeddings[i, :num_dims], marker='o', label=f"Sentence {i+1}")
            
            ax.set_xlabel("Dimension")
            ax.set_ylabel("Value")
            ax.set_title(f"First {num_dims} dimensions of each embedding")
            ax.set_xticks(range(num_dims))
            ax.legend()
            
            st.pyplot(fig)
        
        with tab2:
            st.markdown('<h3 class="task-header">Task Predictions</h3>', unsafe_allow_html=True)
            
            # Define labels
            task_a_labels = ["Negative", "Positive"]  # SST-2 labels
            task_b_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]  # CoNLL-2003 labels
            
            # Display predictions for each sentence
            for i, sentence in enumerate(sentences):
                st.markdown(f"**Sentence {i+1}:** {sentence}")
                
                # Task A prediction (sentiment)
                task_a_pred = results['task_a_preds'][i]
                task_a_probs = results['task_a_probs'][i]
                
                st.markdown(f"**Task A (Sentiment):** <span class='highlight'>{task_a_labels[task_a_pred]}</span>", unsafe_allow_html=True)
                
                # Plot Task A probabilities
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(task_a_labels, task_a_probs)
                ax.set_ylim(0, 1)
                ax.set_title(f"Sentiment Probabilities")
                st.pyplot(fig)
                
                # Task B prediction (NER)
                st.markdown("**Task B (Named Entity Recognition):**")
                
                # Get predictions for tokens
                tokens = sentence.split()
                task_b_pred = results['task_b_preds'][i][:len(tokens)]
                
                # Display tokens with their labels
                html_tokens = []
                for token, label_idx in zip(tokens, task_b_pred):
                    label = task_b_labels[label_idx]
                    html_tokens.append(f"<span class='token-label token-{label}'>{token} ({label})</span>")
                
                st.markdown(" ".join(html_tokens), unsafe_allow_html=True)
                
                st.markdown("---")
        
        with tab3:
            st.markdown('<h3 class="task-header">Similarity Matrix</h3>', unsafe_allow_html=True)
            
            # Plot similarity matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(similarity_matrix, annot=True, cmap='viridis', ax=ax)
            ax.set_xticklabels([f"S{i+1}" for i in range(len(sentences))], rotation=45)
            ax.set_yticklabels([f"S{i+1}" for i in range(len(sentences))])
            ax.set_title("Sentence Similarity Matrix (Cosine Similarity)")
            st.pyplot(fig)
            
            # Print sentence mapping
            st.markdown("**Sentence Mapping:**")
            for i, sentence in enumerate(sentences):
                st.markdown(f"**S{i+1}:** {sentence}")

if __name__ == "__main__":
    main() 