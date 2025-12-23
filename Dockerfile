FROM python:3.11.2

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ /app/src/
COPY saves/ /app/saves/

CMD [ "python3", "./src/app.py" ]
