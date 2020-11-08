# MINT with Docker
MINT is now available on DockerHub in containerized format. A container is a standard unit of software that packages up code and all its dependencies, so the application runs quickly and reliably from one computing environment to another. In contrast to a virtual machine (VM), a Docker container image is a lightweight, standalone, executable package of software that includes everything needed to run an application: code, runtime, system tools, system libraries and settings. This allows to run MINT on any computer that can run Docker.

The following command can be used to pull the latest image from docker hub.

    docker pull msmint/msmint:latest

The image can be started with:

    docker run -p 8501:8501 -it msmint/msmint:latest

Then the tool is available in the browser at http://localhost:8501.
