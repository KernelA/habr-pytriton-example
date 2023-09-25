# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

RUN --mount=type=cache,target=/var/cache/apt \ 
    --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    DEBIAN_FRONTEND=noninteractive apt update && \
    apt install --no-install-recommends python3 libpython3-dev python3-venv python3-pip python-is-python3 curl -y

WORKDIR /home/app

COPY --link ./models/converted ./models/converted

RUN python -m venv /home/app/venv

ENV PATH=/home/app/venv/bin:${PATH}

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    --mount=type=bind,source=./requirements.server.cpu.txt,target=./requirements.txt \
    --mount=type=bind,source=./requirements.server.base.txt,target=./requirements.server.base.txt \
    pip install -r ./requirements.txt

COPY ./server.py ./

ARG HTTP_PORT=8000

ARG METRICS_PORT=8002

ENV PYTRITON_TRITON_CONFIG_HTTP_PORT=${HTTP_PORT} \
    PYTRITON_TRITON_CONFIG_METRICS_PORT=${METRICS_PORT} \
    PYTHONUNBUFFERED=1

EXPOSE ${HTTP_PORT} ${METRICS_PORT}

CMD python ./server.py