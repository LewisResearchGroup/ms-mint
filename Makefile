
publish:
	rm dist/*.*
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload --repository ms-mint dist/ms*mint-*

lint:
	flake8

tests:
	pytest --cov=ms_mint --cov-report html

build:
	pyinstaller --onedir --noconfirm specfiles\Mint__onedir__.spec --additional-hooks-dir=hooks

docs:
	mkdocs build && mkdocs gh-deploy
