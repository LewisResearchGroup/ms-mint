# Contributing to MS-MINT

Thank you for your interest in contributing to MS-MINT.

## Branching Strategy

- **main** - Protected branch for releases only. Direct pushes are blocked.
- **dev** - Active development branch.

## Workflow

1. Create a feature branch from `dev`:
   ```bash
   git checkout dev
   git pull
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit.

3. Push your branch and open a Pull Request to `dev`:
   ```bash
   git push origin feature/your-feature-name
   ```

4. After review, merge to `dev`.

5. For releases, create a PR from `dev` to `main`.

## Code Standards

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) coding standards
- Use **Black** for code formatting
- Use **Flake8** for linting

## Running Tests

```bash
pytest tests/
```

## Documentation

Documentation is built with MkDocs. To preview locally:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

## Reporting Issues

When reporting issues, please include:
- MS-MINT version
- Operating system and Python version
- Steps to reproduce
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the project's license.
