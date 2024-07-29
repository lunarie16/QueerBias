FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

RUN apt update && DEBIAN_FRONTEND="noninteractive" TZ="Europe/Berlin" apt install -y libopenblas-base libaio-dev libomp-dev git-all

RUN mkdir -p /tuning
WORKDIR /tuning


COPY requirements.txt .


RUN pip3 install --upgrade pip setuptools
RUN pip3 install wheel
RUN pip3 install --use-pep517 -r requirements.txt
