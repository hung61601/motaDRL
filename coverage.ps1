conda activate motaDRL
coverage run --omit="tests/*,run_scripts/*" -m pytest
coverage html --skip-empty -d ./tests/coverage
coverage-badge -o coverage.svg -f
Start-Process tests/coverage/index.html