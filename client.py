import pathlib

import gradio as gr
import numpy as np
from tritonclient.http import (InferenceServerClient, InferInput,
                               InferRequestedOutput)


def process_image(image: np.ndarray, model_name: str):
    with InferenceServerClient("localhost:8000") as client:
        model_config = client.get_model_metadata(model_name)

        input_dtype = model_config["inputs"][0]["datatype"]
        input_info = InferInput(model_config["inputs"]
                                [0]["name"], image.shape, input_dtype)

        input_info.set_data_from_numpy(image)

        out_name = model_config["outputs"][0]["name"]
        outputs = [InferRequestedOutput(out_name)]

        infer_res = client.infer(
            model_name,
            inputs=[input_info],
            outputs=outputs
        )

        return infer_res.as_numpy(out_name)


if __name__ == "__main__":
    model_dir = pathlib.Path("models", "converted")

    model_names = tuple(
        map(lambda path: path.stem.split("-")[0], model_dir.glob("*.onnx")))

    with gr.Blocks() as ui:
        with gr.Row():
            in_img = gr.Image(image_mode="RGB", height=300)
            out_img = gr.Image(image_mode="RGB", height=300)

        with gr.Row():
            in_model_names = gr.Radio(model_names, value=model_names[0])
            btn = gr.Button()
            btn.click(process_image, inputs=[
                      in_img, in_model_names], outputs=out_img)

    ui.launch()
