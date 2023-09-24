import pathlib

import onnx


if __name__ == "__main__":
    onnx_models = pathlib.Path("models").glob("*.onnx")
    opset_version = onnx.defs.onnx_opset_version()

    out_dir = pathlib.Path("models", "converted")
    out_dir.mkdir(exist_ok=True, parents=True)

    for onnx_model_path  in onnx_models:
        model = onnx.load(str(onnx_model_path))
        model = onnx.version_converter.convert_version(model, opset_version)
        name = onnx_model_path.stem.split("-")[0]

        onnx.save_model(model, str(out_dir / onnx_model_path.with_stem(f"{name}-{opset_version}").name))

