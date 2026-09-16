FROM archlinux:latest

ENV TERM=xterm

RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
        python \
        python-pip \
        sagemath \
        gap-guava \
        gap-packages \
        base-devel \
        git \
        && pacman -Scc --noconfirm

WORKDIR /app

RUN python -m venv /app/venv --system-site-packages

ENV PATH="/app/venv/bin:$PATH"

RUN pip install --upgrade pip

RUN pip install ldpc

RUN pip install \
    networkx \
    numpy \
    scipy \
    ortools \
    simanneal

COPY . /app

CMD ["python", "examples.py"]