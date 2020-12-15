   
# Installation

### Download release (Windows 10)

For Windows 10 a build is provided [here](https://github.com/soerendip/ms-mint/releases/download/v0.0.30/Mint-0.0.30-Windows10.zip). Simply, unpack the zip-archive and start `Mint.exe`. Then navigate to [http://localhost:9999](http://localhost:9999) if the browser does not open automatically.


### Installation with pip

The program can be installed in a Python 3 (>= 3.7) environment using `pip`:

    pip install ms-mint

Then the GUI can be started with 

    Mint.py

Then navigate to http://localhost:9999 if the browser does not open automatically.


### Installation from source

Here we use `conda` from the [miniconda](https://conda.io/en/latest/miniconda.html) package to install dependencies in a virtual environment.

```bash
git clone https://github.com/soerendip/ms-mint
cd ms-mint

conda create -n ms-mint python=3.8
conda activate ms-mint
pip setup.py install  # for regular install
pip install -e .  # for development
```

### Launching the GUI

The browser based GUI can then be started with

    conda activate ms-mint  # if you run MINT in an Anaconda environment
    Mint.py

Then navigate to the following ULR with your browser: [http://localhost:9999/](http://localhost:9999/). The graphical user interface is explained in more detail [here](gui.md)


# MINT with Docker
MINT is now available on DockerHub in containerized format. A container is a standard unit of software that packages up code and all its dependencies, so the application runs quickly and reliably from one computing environment to another. In contrast to a virtual machine (VM), a Docker container image is a lightweight, standalone, executable package of software that includes everything needed to run an application: code, runtime, system tools, system libraries and settings. This allows to run MINT on any computer that can run Docker.

The following command can be used to pull the latest image from docker hub.

    docker pull msmint/msmint:latest

The image can be started with:

    docker run -p 8501:8501 -it msmint/msmint:latest

Then the tool is available in the browser at http://localhost:8501.

