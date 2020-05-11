FROM python:3.6-slim
COPY ./server.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./active_model_tfidf_SVM.sav /deploy/
COPY ./cert.pem /deploy/
COPY ./key.pem /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80 443 8443 8080
ENTRYPOINT ["python", "server.py"]
