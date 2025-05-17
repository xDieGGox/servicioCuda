FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt-get -qq update && \
    apt-get -qq install -y build-essential python3-pip python3-dev python3-opencv && \
    pip3 install --break-system-packages flask numpy pycuda flask-cors


WORKDIR /app
COPY . .

EXPOSE 5000

CMD ["python3", "app.py"]
