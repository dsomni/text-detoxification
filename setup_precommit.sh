#!/bin/bash
echo "#!/bin/sh
pipenv run pip freeze > requirements.txt
pipenv run black .
pipenv run ruff check . --fix --exit-non-zero-on-fix
git add -A" > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit