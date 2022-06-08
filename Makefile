
publish:
	rm dist/*.*
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload --repository ms-mint dist/ms*mint-*

lint:
	flake8

test:
	pytest --cov=ms_mint --cov-report html
	touch images/coverage.svg && rm images/coverage.svg
	coverage-badge -o images/coverage.svg

pyinstaller:
	cd specfiles && pyinstaller --onedir --noconfirm Mint__onedir__.spec --additional-hooks-dir=hooks

doc:
	cd docs
	make html

deploy:
	rm dist/*
	python setup.py sdist bdist_wheel
	python -m twine upload --repository ms-mint dist/ms*mint-*

	