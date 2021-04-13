FROM python:3.7
EXPOSE 9999
COPY . /app
WORKDIR /app
RUN pip3 install .
RUN mkdir -p /data
CMD Mint.py --data-dir /data
 
