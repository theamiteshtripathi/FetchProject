# Outputs Directory

This directory is used to store model outputs, checkpoints, and evaluation results.

## Directory Structure

- `models/`: Saved model checkpoints
- `logs/`: Training and evaluation logs
- `embeddings/`: Saved sentence embeddings
- `visualizations/`: Visualization outputs

## Usage

When training a model, you can specify the output directory:

```bash
python src/train.py --output_dir outputs/models/my_model
```

When running the model, you can load a saved checkpoint:

```bash
python src/run_model.py --model_path outputs/models/my_model/checkpoint.pt
```

## Note

This directory is automatically created by the `init_project.sh` script. The contents of this directory are not tracked by git (they are in `.gitignore`) to avoid committing large model files to the repository.
