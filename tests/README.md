# Tests

This directory contains test files for the Pratt AI Embeddings utilities.

## Running Tests

To run all tests:
```bash
pytest
```

To run tests with coverage:
```bash
pytest --cov=util --cov-report=html
```

To run specific test files:
```bash
pytest tests/test_qwen_embeddings.py
pytest tests/test_infinity_embeddings.py
```

## Test Structure

- `test_qwen_embeddings.py`: Tests for QwenEmbeddings class
- `test_infinity_embeddings.py`: Tests for InfinityEmbeddings class
- `conftest.py`: Shared test configuration and fixtures

## Writing Tests

When adding new features, please include corresponding tests. Follow the existing patterns:

1. Use pytest fixtures for setup
2. Mock external API calls
3. Test both success and error cases
4. Include type checking tests

## Mock Testing

The tests use mocked API responses to avoid requiring actual API servers during testing. This allows for:

- Fast test execution
- Reliable test results
- Testing error conditions
- CI/CD integration

## Test Coverage

We aim for high test coverage. Check coverage reports after running tests:
```bash
open htmlcov/index.html  # View coverage report
```
