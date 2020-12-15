FROM python:3.7
EXPOSE 9999
COPY . /app
WORKDIR /app
RUN pip3 install .
CMD python scripts/Mint.py
 
