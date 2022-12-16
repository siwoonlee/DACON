FROM pytorchlightning/pytorch_lightning:base-cuda11.3.1-py3.8-torch1.11

RUN apt update && apt install -y \
    build-essential \
    gcc \
    libsasl2-modules  \
    libsasl2-dev \
    vim \
    tmux \
    htop \
    glances

RUN pip install --upgrade pip

WORKDIR /home
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt
