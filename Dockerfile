FROM python:3.7
ENV PYTHONBUFFERED=1
EXPOSE 5000
WORKDIR /app
COPY . /app
RUN pip install -r /app/requirements.txt
#ENTRYPOINT ["python3"]
#CMD ["/app/predict.py"]
