   
# Installation

## Windows Installer

For Windows 10 a build is provided [here](https://github.com/soerendip/ms-mint/releases/latest). 
Simply, unpack the zip-archive and start `Mint.exe`. 
Then navigate to [http://localhost:9999](http://localhost:9999) if the browser does not open automatically.


## With PyPI

The program can be installed in a Python 3 (>= 3.7) environment using `pip`:

    pip install ms-mint

## Launching the GUI

The browser based GUI can then be started with

    Mint.py

Then navigate to the following ULR with your browser: [http://localhost:9999/](http://localhost:9999/).
More information is available in the [GUI](gui.md) section.

MINT frontend. Available optional arguments are:

    -h, --help            show this help message and exit
    --no-browser          do not start the browser
    --version             print current version
    --data-dir DATA_DIR   target directory for MINT data
    --debug               start MINT server in debug mode
    --port PORT           change the port
    --serve-path SERVE_PATH
                            serve app at a different path e.g. '/mint/' to serve
                            the app at 'localhost:9999/mint/' (deprecated)


## Docker
MINT is now available on DockerHub in containerized format. A container is a standard unit of software that packages up code and all its dependencies, so the application runs quickly and reliably from one computing environment to another. In contrast to a virtual machine (VM), a Docker container image is a lightweight, standalone, executable package of software that includes everything needed to run an application: code, runtime, system tools, system libraries and settings. This allows to run MINT on any computer that can run Docker.

The following command can be used to pull the latest image from docker hub.

    docker pull msmint/msmint:latest

The image can be started with:

    docker run -p 9999:9999 -it msmint/msmint:latest

Then the tool is available in the browser at http://localhost:9999.


## From source

Here we use `conda` from the [miniconda](https://conda.io/en/latest/miniconda.html) package to install dependencies in a virtual environment.

```bash
git clone https://github.com/soerendip/ms-mint
cd ms-mint

conda create -n ms-mint python=3.8
conda activate ms-mint
pip setup.py install  # for regular install
pip install -e .  # for development
```