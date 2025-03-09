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
pip install transformers==4.36.2 tokenizers==0.13.3
```

2. Or use the provided script to fix the environment and run the app:

```bash
./fix_and_run.sh
```

### 2. Import Errors in the Frontend

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

### 3. Testing the Streamlit App

If you want to test that Streamlit works correctly without running the full app:

```bash
./run_test_streamlit.sh
```

This will start a simple Streamlit app that tests the imports and verifies that everything is working correctly.

### 4. Conda Environment Issues

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

### 5. Running the Full App

Once you've fixed the environment issues, you can run the full app:

```bash
python src/run_frontend.py
```

Or use the provided script:

```bash
./fix_and_run.sh
```

## Additional Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Sentence Transformers Documentation](https://www.sbert.net/) 