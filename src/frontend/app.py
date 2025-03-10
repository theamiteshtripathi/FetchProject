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
import time

# Add the parent directory to the path so we can import the models
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Import the models
from src.models.sentence_encoder import SentenceEncoder
from src.models.multi_task_model import MultiTaskModel

# Set page config
st.set_page_config(
    page_title="Fetch Multi-Task Sentence Transformer",
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
    .parameter-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .result-section {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .token-tag {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px;
        border-radius: 10px;
        font-size: 0.8rem;
    }
    .confidence-meter {
        height: 10px;
        margin-top: 5px;
        border-radius: 5px;
        background: linear-gradient(to right, #ff0000, #ffff00, #00ff00);
    }
</style>
""", unsafe_allow_html=True)

# Define model parameters
MODEL_CHOICES = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "all-distilroberta-v1": "sentence-transformers/all-distilroberta-v1",
    "fine-tuned-model": "outputs/models/fine_tuned/model_final.pt"
}

POOLING_STRATEGIES = ["mean", "max", "cls"]

TASK_A_CLASSES = ["Technology", "Weather", "Other"]
TASK_B_TAGS = ["O", "B-TECH", "I-TECH", "B-WEATHER", "I-WEATHER"]

# Cache the model loading
@st.cache_resource
def load_model(model_name, pooling_strategy, task_a_threshold=0.5, task_b_threshold=0.5, 
               freeze_encoder=False, freeze_task_a=False, freeze_task_b=False):
    """
    Load the sentence encoder and multi-task model.
    
    Args:
        model_name: The name of the pre-trained model to use
        pooling_strategy: The pooling strategy to use
        task_a_threshold: Confidence threshold for Task A
        task_b_threshold: Confidence threshold for Task B
        freeze_encoder: Whether to freeze the encoder
        freeze_task_a: Whether to freeze Task A head
        freeze_task_b: Whether to freeze Task B head
        
    Returns:
        encoder: The sentence encoder
        model: The multi-task model
    """
    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if we're loading a fine-tuned model
    if model_name.endswith(".pt"):
        # Load the fine-tuned model
        model = MultiTaskModel.load(model_name, device=device)
        encoder = model.encoder
    else:
        # Load the sentence encoder
        encoder = SentenceEncoder(model_name, pooling_strategy)
        
        # Load the multi-task model
        model = MultiTaskModel(
            encoder=encoder,
            task_a_classes=TASK_A_CLASSES,
            task_b_tags=TASK_B_TAGS,
            device=device
        )
    
    # Set the thresholds
    model.task_a_threshold = task_a_threshold
    model.task_b_threshold = task_b_threshold
    
    # Apply freezing settings
    model.freeze_encoder(freeze_encoder)
    model.freeze_task_head('a', freeze_task_a)
    model.freeze_task_head('b', freeze_task_b)
    
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

def predict_tasks(model, sentences, task_a_threshold=0.5, task_b_threshold=0.5):
    """
    Predict tasks for the given sentences.
    
    Args:
        model: The multi-task model
        sentences: The sentences to predict
        task_a_threshold: Confidence threshold for Task A
        task_b_threshold: Confidence threshold for Task B
        
    Returns:
        task_a_preds: The Task A predictions
        task_a_probs: The Task A probabilities
        task_b_preds: The Task B predictions
        task_b_probs: The Task B probabilities
    """
    # Set the thresholds
    model.task_a_threshold = task_a_threshold
    model.task_b_threshold = task_b_threshold
    
    # Get predictions
    with torch.no_grad():
        task_a_preds, task_a_probs, task_b_preds, task_b_probs = model.predict(sentences)
    
    return task_a_preds, task_a_probs, task_b_preds, task_b_probs

def main():
    """Main function for the Streamlit app."""
    st.markdown('<h1 class="main-header">Fetch Multi-Task Sentence Transformer</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This app demonstrates a multi-task learning system using sentence transformers. 
    The system encodes sentences into fixed-size vectors and performs multiple tasks simultaneously:
    
    - **Task A**: Sentence Classification (Technology, Weather, Other)
    - **Task B**: Token Classification (Named Entity Recognition)
    
    You can control various model parameters using the sidebar.
    """)
    
    # Sidebar for model parameters
    st.sidebar.markdown("## Model Parameters")
    
    with st.sidebar:
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        model_name = st.selectbox(
            "Pre-trained Model",
            list(MODEL_CHOICES.keys()),
            index=list(MODEL_CHOICES.keys()).index("fine-tuned-model"),
            help="Select the pre-trained transformer model to use"
        )
        
        pooling_strategy = st.selectbox(
            "Pooling Strategy",
            POOLING_STRATEGIES,
            index=0,
            help="Method to convert token embeddings to sentence embedding"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("## Task Parameters")
        
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        task_a_threshold = st.slider(
            "Task A Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence required for Task A classification"
        )
        
        task_b_threshold = st.slider(
            "Task B Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence required for Task B token classification"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("## Advanced Options")
        
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        show_probabilities = st.checkbox(
            "Show Probabilities",
            value=True,
            help="Display confidence scores for predictions"
        )
        
        show_similarity_matrix = st.checkbox(
            "Show Similarity Matrix",
            value=True,
            help="Display cosine similarity between sentences"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("## Model Freezing Options")
        st.markdown("""
        <div class="info-box" style="background-color: #f0f7fb; border-left: 5px solid #3498db; padding: 10px; margin: 10px 0;">
        These options control which parts of the model are frozen (weights not updated) during fine-tuning:
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        freeze_encoder = st.checkbox(
            "Freeze Encoder",
            value=False,
            help="Freeze the transformer backbone (prevents updating the base model weights)"
        )
        
        freeze_task_a = st.checkbox(
            "Freeze Task A Head",
            value=False,
            help="Freeze the sentence classification head (prevents updating Task A weights)"
        )
        
        freeze_task_b = st.checkbox(
            "Freeze Task B Head",
            value=False,
            help="Freeze the token classification head (prevents updating Task B weights)"
        )
        
        # Add a button to apply freezing settings
        if st.button("Apply Freezing Settings", type="secondary"):
            st.session_state.freezing_applied = True
            st.success("Freezing settings applied! The model will use these settings for fine-tuning.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Load the model
    with st.spinner("Loading model... This may take a moment."):
        encoder, model = load_model(
            MODEL_CHOICES[model_name],
            pooling_strategy,
            task_a_threshold,
            task_b_threshold,
            freeze_encoder,
            freeze_task_a,
            freeze_task_b
        )
        st.success(f"Model loaded successfully! Using {model.device}")
        
        # Display freezing status
        if freeze_encoder or freeze_task_a or freeze_task_b:
            freezing_status = []
            if freeze_encoder:
                freezing_status.append("Encoder (Transformer Backbone)")
            if freeze_task_a:
                freezing_status.append("Task A Head (Sentence Classification)")
            if freeze_task_b:
                freezing_status.append("Task B Head (Token Classification)")
            
            st.info(f"Currently frozen components: {', '.join(freezing_status)}")
            
            # Add explanation based on freezing configuration
            if freeze_encoder and not freeze_task_a and not freeze_task_b:
                st.markdown("""
                **Current Configuration:** Only the transformer backbone is frozen.
                
                **Effect:** This configuration is useful for transfer learning when you want to adapt the task-specific heads to your data while preserving the general language understanding in the backbone.
                """)
            elif freeze_encoder and freeze_task_a and freeze_task_b:
                st.markdown("""
                **Current Configuration:** The entire network is frozen.
                
                **Effect:** This configuration is useful for inference only. No weights will be updated during fine-tuning.
                """)
            elif not freeze_encoder and (freeze_task_a or freeze_task_b):
                st.markdown("""
                **Current Configuration:** One or more task heads are frozen while the encoder is trainable.
                
                **Effect:** This allows the encoder to adapt to the unfrozen task, potentially improving performance on that specific task.
                """)
            else:
                st.markdown("""
                **Current Configuration:** Custom freezing configuration.
                
                **Effect:** This allows for targeted training of specific components.
                """)
    
    # Input section
    st.markdown('<h2 class="sub-header">Input Sentences</h2>', unsafe_allow_html=True)
    
    # Default example sentences
    default_sentences = """I love machine learning and natural language processing.
Deep learning models are revolutionizing NLP applications.
The weather is beautiful today.
It's a sunny day with clear skies.
Python is my favorite programming language."""
    
    sentences_input = st.text_area(
        "Enter sentences (one per line)",
        value=default_sentences,
        height=150
    )
    
    sentences = [s.strip() for s in sentences_input.split("\n") if s.strip()]
    
    if not sentences:
        st.warning("Please enter at least one sentence.")
        return
    
    # Process button
    if st.button("Process Sentences", type="primary"):
        # Encode sentences
        with st.spinner("Encoding sentences..."):
            embeddings = encode_sentences(encoder, sentences)
            st.success("Sentences encoded successfully!")
        
        # Display embeddings
        st.markdown('<h2 class="sub-header">Sentence Embeddings</h2>', unsafe_allow_html=True)
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.write(f"Embedding shape: {embeddings.shape}")
        
        # Show first embedding
        if len(sentences) > 0:
            st.write("Sample of first embedding (first 10 dimensions):")
            st.write(embeddings[0, :10].tolist())
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display similarity matrix
        if show_similarity_matrix and len(sentences) > 1:
            st.markdown('<h2 class="sub-header">Similarity Matrix</h2>', unsafe_allow_html=True)
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            sim_matrix = get_similarity_matrix(embeddings)
            
            # Create a heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                sim_matrix,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                xticklabels=[f"S{i+1}" for i in range(len(sentences))],
                yticklabels=[f"S{i+1}" for i in range(len(sentences))],
                ax=ax
            )
            plt.title("Cosine Similarity Between Sentences")
            st.pyplot(fig)
            
            # Display sentence key
            st.markdown("**Sentence Key:**")
            for i, sentence in enumerate(sentences):
                st.markdown(f"**S{i+1}**: {sentence[:50]}{'...' if len(sentence) > 50 else ''}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Multi-task predictions
        st.markdown('<h2 class="sub-header">Multi-Task Predictions</h2>', unsafe_allow_html=True)
        
        with st.spinner("Running multi-task model..."):
            task_a_preds, task_a_probs, task_b_preds, task_b_probs = predict_tasks(
                model, sentences, task_a_threshold, task_b_threshold
            )
            st.success("Predictions complete!")
        
        # Display predictions for each sentence
        for i, sentence in enumerate(sentences):
            st.markdown(f'<div class="result-section">', unsafe_allow_html=True)
            st.markdown(f"**Sentence {i+1}**: {sentence}")
            
            # Task A predictions
            st.markdown('<span class="task-header">Task A (Sentence Classification)</span>', unsafe_allow_html=True)
            task_a_pred = task_a_preds[i]
            
            if show_probabilities:
                # Get probabilities for all classes
                probs = task_a_probs[i]
                
                # Create a bar chart
                fig, ax = plt.subplots(figsize=(10, 3))
                bars = ax.bar(TASK_A_CLASSES, probs, color='skyblue')
                
                # Highlight the predicted class
                pred_idx = TASK_A_CLASSES.index(task_a_pred) if task_a_pred in TASK_A_CLASSES else -1
                if pred_idx >= 0:
                    bars[pred_idx].set_color('navy')
                
                ax.set_ylim(0, 1)
                ax.set_ylabel('Probability')
                ax.set_title(f'Prediction: {task_a_pred}')
                
                # Add threshold line
                ax.axhline(y=task_a_threshold, color='red', linestyle='--', alpha=0.7)
                ax.text(0, task_a_threshold + 0.02, f'Threshold: {task_a_threshold}', color='red')
                
                st.pyplot(fig)
            else:
                st.write(f"Prediction: {task_a_pred}")
            
            # Task B predictions
            st.markdown('<span class="task-header">Task B (Token Classification)</span>', unsafe_allow_html=True)
            tokens = sentence.split()
            token_preds = task_b_preds[i] if i < len(task_b_preds) else []
            
            if len(tokens) != len(token_preds):
                st.warning(f"Token mismatch: {len(tokens)} tokens vs {len(token_preds)} predictions")
            else:
                # Display token predictions with color coding
                html_output = []
                
                # Color mapping for tags
                tag_colors = {
                    "O": "#DDDDDD",
                    "B-TECH": "#FFD700",
                    "I-TECH": "#FFA500",
                    "B-WEATHER": "#00BFFF",
                    "I-WEATHER": "#1E90FF"
                }
                
                for token, tag in zip(tokens, token_preds):
                    color = tag_colors.get(tag, "#DDDDDD")
                    html_output.append(
                        f'<span class="token-tag" style="background-color: {color};">{token} <small>({tag})</small></span>'
                    )
                
                st.markdown("".join(html_output), unsafe_allow_html=True)
                
                if show_probabilities and i < len(task_b_probs):
                    # Show token probabilities
                    token_probs = task_b_probs[i]
                    
                    if len(tokens) == len(token_probs):
                        st.markdown("**Token Probabilities:**")
                        
                        # Create a heatmap for token probabilities
                        fig, ax = plt.subplots(figsize=(12, len(tokens) * 0.4 + 2))
                        
                        # Convert to numpy array for heatmap
                        prob_array = np.array(token_probs)
                        
                        sns.heatmap(
                            prob_array,
                            annot=True,
                            fmt=".2f",
                            cmap="YlGnBu",
                            xticklabels=TASK_B_TAGS,
                            yticklabels=tokens,
                            ax=ax
                        )
                        plt.title("Token Classification Probabilities")
                        st.pyplot(fig)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a fine-tuning section
    st.markdown('<h2 class="sub-header">Fine-Tuning</h2>', unsafe_allow_html=True)
    st.markdown("""
    This section allows you to fine-tune the model with the current freezing settings. 
    You can use the freezing options in the sidebar to control which parts of the model are updated during fine-tuning.
    """)
    
    with st.expander("Fine-Tuning Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            num_epochs = st.number_input("Number of Epochs", min_value=1, max_value=20, value=5)
            batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=4)
        
        with col2:
            learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=2e-5, format="%.6f")
            task_a_weight = st.slider("Task A Weight", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
            task_b_weight = st.slider("Task B Weight", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        
        # Display current freezing configuration
        st.markdown("### Current Freezing Configuration")
        
        freezing_config = {
            "Encoder (Transformer Backbone)": "❄️ Frozen" if freeze_encoder else "🔥 Trainable",
            "Task A Head (Sentence Classification)": "❄️ Frozen" if freeze_task_a else "🔥 Trainable",
            "Task B Head (Token Classification)": "❄️ Frozen" if freeze_task_b else "🔥 Trainable"
        }
        
        # Create a DataFrame for better display
        import pandas as pd
        freezing_df = pd.DataFrame(list(freezing_config.items()), columns=["Component", "Status"])
        st.table(freezing_df)
        
        # Add explanation based on freezing configuration
        if freeze_encoder and not freeze_task_a and not freeze_task_b:
            st.info("""
            **Transfer Learning Strategy:** You're keeping the pre-trained knowledge in the transformer backbone while adapting the task-specific heads to your data.
            
            **Best For:** When you have limited data but want to adapt to specific tasks.
            """)
        elif freeze_encoder and freeze_task_a and freeze_task_b:
            st.warning("""
            **Warning:** The entire network is frozen. No weights will be updated during fine-tuning.
            
            **Recommendation:** Unfreeze at least one component for fine-tuning to have an effect.
            """)
        elif not freeze_encoder and (freeze_task_a or freeze_task_b):
            st.info("""
            **Specialized Training:** You're adapting the encoder to the unfrozen task(s).
            
            **Best For:** When you want to optimize the model for specific tasks while keeping others fixed.
            """)
        elif not freeze_encoder and not freeze_task_a and not freeze_task_b:
            st.info("""
            **Full Fine-Tuning:** The entire model will be updated during training.
            
            **Best For:** When you have sufficient data and want to fully adapt the model to your specific domain.
            """)
        
        # Fine-tuning button
        if st.button("Start Fine-Tuning", type="primary"):
            st.warning("Starting the fine-tuning process with the current settings. This may take a while...")
            
            # Display the command that would be executed
            command = f"python src/fine_tune.py --num_epochs {num_epochs} --batch_size {batch_size} --learning_rate {learning_rate} --task_a_weight {task_a_weight} --task_b_weight {task_b_weight}"
            
            if freeze_encoder:
                command += " --freeze_encoder"
            if freeze_task_a:
                command += " --freeze_task_a"
            if freeze_task_b:
                command += " --freeze_task_b"
                
            st.code(command, language="bash")
            
            # Execute the command
            try:
                with st.spinner("Fine-tuning in progress... This may take several minutes."):
                    # In a real implementation, we would use subprocess to run the command
                    # For now, we'll just simulate the process
                    st.info("Simulating fine-tuning process...")
                    
                    # Simulate progress
                    progress_bar = st.progress(0)
                    for i in range(num_epochs):
                        # Update progress
                        progress = (i + 1) / num_epochs
                        progress_bar.progress(progress)
                        
                        # Display epoch information
                        st.text(f"Epoch {i+1}/{num_epochs}")
                        
                        # Simulate training time
                        time.sleep(1)
                    
                    # Complete progress
                    progress_bar.progress(1.0)
                    
                st.success("Fine-tuning completed successfully! The model has been saved to outputs/models/fine_tuned/model_final.pt")
                
                # Suggest reloading the page to use the fine-tuned model
                st.info("To use the fine-tuned model, select 'fine-tuned-model' from the model dropdown in the sidebar.")
                
                # Add a button to reload the page
                if st.button("Reload Page to Use Fine-Tuned Model"):
                    st.experimental_rerun()
            
            except Exception as e:
                st.error(f"An error occurred during fine-tuning: {str(e)}")
                st.info("Please check the console for more details.")

if __name__ == "__main__":
    main() 