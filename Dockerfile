FROM python:3.7
EXPOSE 9999

RUN mkdir -p /data

COPY requirements.txt .
COPY local_wheels/dash_uploader-0.5.0-py3-none-any.whl /local_wheels/dash_uploader-0.5.0-py3-none-any.whl

RUN pip3 install -r requirements.txt

COPY . /app
WORKDIR /app
RUN pip3 install .

ENV MINT_DATA_DIR /data
ENV DATABASE_URL sqlite:////data/mint.db

RUN pip install local_wheels/dash_uploader-0.5.0-py3-none-any.whl 

RUN echo $DATABASE_URL

CMD ./entrypoint.sh

