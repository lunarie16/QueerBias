FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

RUN apt update && DEBIAN_FRONTEND="noninteractive" TZ="Europe/Berlin" apt install -y libopenblas-base libomp-dev git-all

RUN mkdir -p /tuning
WORKDIR /tuning

COPY requirements.txt .

COPY code ./code-local
#COPY data/datasets ./data/datasets

RUN pip3 install --upgrade pip setuptools
RUN pip3 install wheel
RUN pip3 cache purge
RUN pip3 install --use-pep517 -r requirements.txt
RUN #python -m spacy download en_core_web_sm