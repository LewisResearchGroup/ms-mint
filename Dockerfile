FROM python:3.7
EXPOSE 9999

RUN mkdir -p /data

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . /app
WORKDIR /app
RUN pip3 install .

ENV MINT_DATA_DIR /data
ENV DATABASE_URL sqlite:///data/mint.db

CMD ./entrypoint.sh
 
