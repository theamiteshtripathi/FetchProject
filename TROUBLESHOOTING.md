# Troubleshooting Guide

## Common Issues and Solutions

### 1. Transformers and Tokenizers Version Conflict

**Issue**: The Streamlit app fails with an error related to incompatible versions of `transformers` and `tokenizers`:

```
ImportError: tokenizers>=0.11.1,!=0.11.3,<0.14 is required for a normal functioning of this module, but found tokenizers==0.19.1.
```

**Solution**:

1. Update your environment with compatible versions:

```bash
pip install transformers==4.36.2 tokenizers==0.21.0
```

2. Or use the provided script to fix the environment and run the app:

```bash
./fix_and_run.sh
```

### 2. PyTorch and Streamlit Conflict

**Issue**: The Streamlit app fails with errors related to PyTorch:

```
RuntimeError: no running event loop

RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_
```

**Solution**:

1. Use the provided script that sets an environment variable to prevent Streamlit from watching PyTorch modules:

```bash
./run_streamlit.sh
```

2. Or set the environment variable manually before running the app:

```bash
export STREAMLIT_WATCH_EXCLUDE_MODULES="torch,tensorflow"
streamlit run src/frontend/app.py
```

### 3. Import Errors in the Frontend

**Issue**: The app fails with import errors when trying to import models.

**Solution**:

1. Make sure the import paths are correct. The imports should be from `src.models` instead of just `models`.

2. Run the test script to verify that the imports work correctly:

```bash
python test_imports.py
```

3. If the test fails, run the fix script:

```bash
./fix_and_run.sh
```

### 4. Testing the Streamlit App

If you want to test that Streamlit works correctly without running the full app:

```bash
./run_test_streamlit.sh
```

This will start a simple Streamlit app that tests the imports and verifies that everything is working correctly.

### 5. Conda Environment Issues

If you're having issues with the conda environment:

1. Update the environment with the latest dependencies:

```bash
conda env update -f environment.yml
```

2. Activate the environment:

```bash
conda activate fetch
```

3. Verify that the environment is correctly set up:

```bash
python test_imports.py
```

### 6. Running the Full App

Once you've fixed the environment issues, you can run the full app:

```bash
# Run the frontend
./run_streamlit.sh

# Run the API
python src/run_api.py
```

Or use Docker:

```bash
# Run the API
./docker/run_docker.sh --mode api

# Run the frontend
./docker/run_docker.sh --mode frontend
```

## Additional Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
