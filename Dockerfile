FROM python:3.8

RUN mkdir /poc_chatbot

WORKDIR /poc_chatbot

COPY . /poc_chatbot

RUN pip install -r requirements.txt

