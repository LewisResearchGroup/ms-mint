# Developer Notes

    python3 setup.py sdist bdist_wheel
    python3 -m twine upload --repository ms-mint dist/ms*mint-*

### On windows
    pyinstaller --onedir --noconfirm specfiles\Mint__onedir__.spec

### Docs

    mkdocs build && mkdocs gh-deploy