#!/usr/bin/env python3
"""Export the YOLOv8 gesture classifier for an ARTPEC-8 or ARTPEC-9 package."""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=("a8", "a9"))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--calibration-images", type=int, default=300)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--labels", type=Path, default=Path("app/model/gesture-labels.txt"))
    return parser.parse_args()


def calibration_images(directory: Path, limit: int):
    images = sorted(
        path for extension in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        for path in directory.rglob(extension)
    )[:limit]
    if not images:
        raise RuntimeError(f"No calibration images found in {directory}")
    return images


def verify_labels(path: Path, names):
    expected = [names[index] for index in range(len(names))]
    actual = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if actual != expected:
        raise RuntimeError(f"{path} does not match the {len(expected)} checkpoint classes")


def convert(weights: str, output: Path, images, size: int, target: str):
    model = YOLO(weights)
    onnx_path = Path(model.export(format="onnx", imgsz=size, opset=12, simplify=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_path = Path(temporary_dir)
        sample_path = temporary_path / "calibration_image_sample_data_20x128x128x3_float32.npy"
        saved_model = temporary_path / "gesture_saved_model"
        np.save(sample_path, np.zeros((20, 128, 128, 3), dtype=np.float32))
        subprocess.run(
            ["onnx2tf", "-i", str(onnx_path.resolve()), "-o", str(saved_model), "-osd", "-n"],
            check=True,
            cwd=temporary_dir,
        )

        def representative_dataset():
            for image_path in images:
                image = tf.io.decode_image(tf.io.read_file(str(image_path)), channels=3,
                                           expand_animations=False)
                image = tf.image.resize(image, [size, size], antialias=True)
                image = tf.cast(image, tf.float32) / 255.0
                yield [tf.expand_dims(image, 0)]

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        if target == "a8":
            converter._experimental_disable_per_channel = True
        output.write_bytes(converter.convert())
    return len(model.names)


def verify(output: Path, size: int, classes: int, target: str):
    interpreter = tf.lite.Interpreter(model_path=str(output))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if tuple(input_detail["shape"]) != (1, size, size, 3) or input_detail["dtype"] is not np.uint8:
        raise RuntimeError(f"Unexpected gesture input: {input_detail['shape']} {input_detail['dtype']}")
    if tuple(output_detail["shape"]) != (1, classes) or output_detail["dtype"] is not np.uint8:
        raise RuntimeError(f"Unexpected gesture output: {output_detail['shape']} {output_detail['dtype']}")

    per_axis = [detail["name"] for detail in interpreter.get_tensor_details()
                if len(detail["quantization_parameters"]["scales"]) > 1]
    if target == "a8" and per_axis:
        raise RuntimeError(f"ARTPEC-8 gesture export contains {len(per_axis)} per-axis tensors")
    if target == "a9" and not per_axis:
        raise RuntimeError("ARTPEC-9 gesture export contains no per-axis tensors")
    print(f"Verified {output}: input 1x{size}x{size}x3, output [1,{classes}], "
          f"per-axis tensors={len(per_axis)}")


def main():
    args = parse_args()
    output = args.output or Path(f"app/model/gesture-{args.target}.tflite")
    model = YOLO(args.weights)
    verify_labels(args.labels, model.names)
    images = calibration_images(args.calibration_dir, args.calibration_images)
    classes = convert(args.weights, output, images, args.size, args.target)
    verify(output, args.size, classes, args.target)


if __name__ == "__main__":
    main()