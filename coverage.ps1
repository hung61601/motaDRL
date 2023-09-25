conda activate motaDRL
coverage run --omit="*test.py" -m pytest
coverage html --skip-empty -d .\tests\coverage
coverage-badge -o coverage.svg -f
Start-Process tests\coverage\index.html