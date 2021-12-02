FROM python:3.7

COPY ./app/ .

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /app

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]