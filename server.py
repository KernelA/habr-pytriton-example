import pathlib
from typing import NamedTuple, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from pytriton.decorators import sample
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig


class PadInfo(NamedTuple):
    orig_image_width: int
    orig_image_height: int


def resize_image(image: np.ndarray, max_size: int):
    height, width = image.shape[:2]

    aspect_ratio = width / height

    if width > height:
        new_width = max_size
        new_height = min(round(new_width / aspect_ratio), max_size)
    else:
        new_width = min(round(max_size * aspect_ratio), max_size)
        new_height = max_size

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)


def pad_image(image: np.ndarray, required_width: int, required_height: int) -> Tuple[np.ndarray, PadInfo]:
    pad_width = required_width - image.shape[1]
    pad_height = required_height - image.shape[0]
    return np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode="constant"), PadInfo(image.shape[1], image.shape[0])


class StyleTransferONNX:
    def __init__(self, path_to_model: str, out_api_name: str):
        self._session = ort.InferenceSession(
            path_to_model, providers=ort.get_available_providers())
        input_info = self._session.get_inputs()[0]
        self._input_name = input_info.name
        self._in_height, self._in_width = input_info.shape[2:]
        self._max_image_size = min(self._in_height, self._in_width)
        self._out_model_name = self._session.get_outputs()[0].name
        self._out_api_name = out_api_name

    @sample
    def __cal__(self, image: np.ndarray):
        orig_height, orig_width = image.shape[:2]
        image = resize_image(image, self._max_image_size)
        image, pad_info = pad_image(
            image, self._in_width, self._in_height)
        image = image.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        out_image_batch = self._session.run(
            [self._out_model_name], {self._input_name: image})[0]

        out_image = out_image_batch[0].transpose(1, 2, 0)[:pad_info.orig_image_height,
                                                          :pad_info.orig_image_width, ...]
        out_image = out_image.clip(0, 255).astype(np.uint8)

        out_image = cv2.resize(out_image, (orig_width, orig_height),
                               interpolation=cv2.INTER_LANCZOS4)

        return {self._out_api_name: out_image}


if __name__ == "__main__":
    model_dir = pathlib.Path("models", "converted")
    onnx_model_paths = list(model_dir.glob("*.onnx"))
    assert len(onnx_model_paths) > 0, "Cannot find any ONNX model"

    with Triton(config=TritonConfig(strict_readiness=True)) as triton:
        out_api_name = "styled_image"

        for onnx_path in onnx_model_paths:
            model = StyleTransferONNX(str(onnx_path), out_api_name)
            model_name = onnx_path.stem.split("-")[0]

            triton.bind(
                model_name=model_name,
                infer_func=model.__cal__,
                inputs=[
                    Tensor(dtype=np.uint8, shape=(-1, -1, 3), name="image"),
                ],
                outputs=[
                    Tensor(dtype=np.uint8, shape=(-1, -1, 3),
                           name=out_api_name),
                ],
                config=ModelConfig(batching=False)
            )

        triton.serve()
