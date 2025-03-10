# Tests

This directory contains tests for the Sentence Transformer Multi-Task Learning project.

## Running Tests

To run all tests:

```bash
python -m unittest discover tests
```

To run a specific test file:

```bash
python -m unittest tests/test_api.py
```

## Test Files

- `test_api.py`: Tests for the API endpoints

## Adding New Tests

When adding new tests, follow these guidelines:

1. Create a new file named `test_*.py` for each module you want to test
2. Use the `unittest` framework for writing tests
3. Follow the pattern of existing tests
4. Ensure tests are independent and can run in any order
5. Use meaningful test names that describe what is being tested 