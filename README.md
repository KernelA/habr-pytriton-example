# Pytriton with Gradio

A simple example of using [pytriton inference server](https://github.com/triton-inference-server/pytriton) with [Gradio](https://www.gradio.app/).


## Requirements

1. [NVIDIA Container Toolkit (optional)](https://github.com/NVIDIA/nvidia-docker)
2. Docker compose v2.
3. Python 3.8 or higher.

## How to run

### Prepare models

```
pip install  -r ./requirements.base.txt
```

```
dvc pull
dvc repro
```

### Server

Run pytriton inference server:
```
docker compose up --build -d
```

Check status. It must be (healthy).
```
docker compose ps
```

If you want to use GPU uncomment `deploy` section in docker compose and replace in the `Dockerfile`:
```
--mount=type=bind,source=./requirements.server.cpu.txt,target=./requirements.txt \
```
to
```
--mount=type=bind,source=./requirements.server.gpu.txt,target=./requirements.txt \
```

### Client

Run gradio app:
```
pip install -r ./requirements.client.txt
```

```
python ./client.py
```

Open UI by link.
