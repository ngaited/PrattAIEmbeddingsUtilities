# Contributing to Pratt AI Embeddings

Thank you for your interest in contributing to the Pratt AI Embeddings project! We welcome contributions from the community.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- A GitHub account

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/pratt-ai-embeddings.git
   cd pratt-ai-embeddings
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Guidelines

### Code Style

We use [Black](https://black.readthedocs.io/) for code formatting and [flake8](https://flake8.pycqa.org/) for linting.

- Run Black: `black .`
- Run flake8: `flake8 util/`
- Run type checking: `mypy util/`

### Testing

We use [pytest](https://pytest.org/) for testing. Please write tests for new features and ensure existing tests pass.

- Run tests: `pytest`
- Run tests with coverage: `pytest --cov=util`

### Documentation

- Update the README.md if you add new features
- Add docstrings to all public functions and classes
- Follow Google-style docstring format

## Submitting Changes

### Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a Pull Request on GitHub

### Pull Request Guidelines

- **Clear description**: Explain what your PR does and why
- **Small, focused changes**: Keep PRs small and focused on a single feature or fix
- **Tests**: Include tests for new functionality
- **Documentation**: Update documentation as needed
- **Code quality**: Ensure code passes all linting and type checking

### Commit Message Format

Use clear, descriptive commit messages:

```
feat: add support for new embedding model
fix: resolve timeout issues in batch processing
docs: update API reference for rerank method
test: add unit tests for similarity computation
```

## Types of Contributions

### Bug Reports

When filing a bug report, please include:

- A clear description of the issue
- Steps to reproduce the problem
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Code samples or error messages

### Feature Requests

For feature requests, please include:

- A clear description of the desired feature
- Use cases and rationale
- Possible implementation approaches
- Any related issues or discussions

### Code Contributions

We welcome contributions in these areas:

- **New embedding providers**: Support for additional embedding APIs
- **Performance improvements**: Optimizations for batch processing
- **Error handling**: Better error messages and recovery
- **Documentation**: Examples, tutorials, and API documentation
- **Testing**: Unit tests, integration tests, and benchmarks

## API Design Principles

When contributing to the codebase, please follow these principles:

1. **LangChain Compatibility**: Maintain compatibility with LangChain interfaces
2. **Consistency**: Follow existing patterns and naming conventions
3. **Error Handling**: Provide clear error messages and graceful degradation
4. **Type Safety**: Use proper type hints throughout
5. **Documentation**: Include comprehensive docstrings and examples

## Review Process

All submissions require review. Here's what to expect:

1. **Automated checks**: CI will run tests, linting, and type checking
2. **Code review**: A maintainer will review your code for quality and design
3. **Feedback**: You may receive requests for changes or improvements
4. **Approval**: Once approved, your PR will be merged

## Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Focus on constructive feedback
- Follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)

## Questions?

If you have questions about contributing, please:

- Check existing issues and discussions
- Open a new issue with the "question" label
- Contact the maintainers

Thank you for contributing to Pratt AI Embeddings!
