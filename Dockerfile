FROM python:3.7
EXPOSE 9999
WORKDIR /app
ADD requirements.txt .
RUN pip3 install -r requirements.txt
ADD ./scripts /app
CMD python Mint.py
 
