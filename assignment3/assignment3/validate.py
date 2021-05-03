import sys

from flake8.main.application import Application
from importlinter.cli import lint_imports, EXIT_STATUS_SUCCESS
import pytest

if __name__ == "__main__":
    # Lint code
    flake8 = Application()
    flake8.run([
        'assignment',
        '--max-line-length', '100',
    ])

    if flake8.result_count > 0:
        print('-- flake8 found code style issues --')
        flake8.exit()

    if (exit_code := lint_imports()) != EXIT_STATUS_SUCCESS:
        print('-- import-linter found invalid packages --')
        sys.exit(exit_code)

    # Run tests
    pytest.main(['-v', '--basetemp', 'processing_results', '-W', 'ignore::DeprecationWarning'])
