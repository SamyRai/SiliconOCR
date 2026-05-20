# Contributing

## Setup

1. Clone the repository.
2. Install dependencies via `uv`:
   ```bash
   make install
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```

## Pull Requests

1. Format and lint code:
   ```bash
   make format
   make lint
   ```
2. Run tests:
   ```bash
   make test
   ```
3. Alternatively, run `make check` to run all validation.
4. Update `README.md` if the API or behavior changes.
5. Submit the PR using the provided template.
