FROM python:3.10-slim-buster AS base

RUN apt update && apt install -y \
    build-essential \
    gcc \
    libsasl2-modules  \
    libsasl2-dev \
    vim \
    tmux \
    htop

WORKDIR /home
RUN pip3 install auto-sklearn
