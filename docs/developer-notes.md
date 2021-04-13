# Developer Notes
    python3 setup.py sdist bdist_wheel
    python3 -m twine upload --repository ms-mint dist/ms*mint-*

## Windows executables
    pyinstaller --onedir --noconfirm specfiles\Mint__onedir__.spec --additional-hooks-dir=hooks

## Documentation deployment

    mkdocs build && mkdocs gh-deploy

## Example NGINX config

    location /mint/ {
        client_max_body_size    10G;
        proxy_pass       http://localhost:9999;
        #rewrite ^/mint/(.*) /$1 break;

        proxy_http_version 1.1;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }

Then start MINT with `--serve-path='\mint\'`.


## Additional packages

To run tests and code optimization you need the
following packages:

    conda install flake8 pytest mkdocs