

prettier:
	uv run black --skip-string-normalization .

test:
	for env in `uvx tox -l`; do echo $$env; uvx tox -e $$env || break; done