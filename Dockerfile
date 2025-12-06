FROM python:3.11.2

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ /app/src/
COPY data/ /app/data/
COPY test.txt .

RUN mkdir /app/saves/

RUN [ "python3", "./src/train.py" ]
CMD [ "python3", "./src/predict.py" ]
