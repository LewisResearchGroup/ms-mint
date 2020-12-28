# Developer Notes
    python3 setup.py sdist bdist_wheel
    python3 -m twine upload --repository ms-mint dist/ms*mint-*

## Windows executables
    pyinstaller --onedir --noconfirm specfiles\Mint__onedir__.spec --additional-hooks-dir=hooks

## Documentation deployment

    mkdocs build && mkdocs gh-deploy