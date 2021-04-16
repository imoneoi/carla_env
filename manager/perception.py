import json

import numpy as np
import onnxruntime


class PerceptionManager:
    def __init__(self,
                 global_options: dict,
                 gpu_index: int):
        self.options = {
            "drivable_model": "static_files/models/ep_44_iou_0.9343.checkpoint.onnx",
            "human_readable": False
        }
        self.options.update(global_options.get("perception", {}))

        print("Perception Manager Options: ", json.dumps(self.options))

        # load models
        self.drivable_model = onnxruntime.InferenceSession(self.options["drivable_model"])

    def infer(self, images):
        # imagenet normalization
        dtype = np.float32
        x = np.array(images, dtype=dtype) / 255
        x = (x - np.array([0.485, 0.456, 0.406], dtype=dtype)) / np.array([0.229, 0.224, 0.225], dtype=dtype)

        # split batch
        num_cars, num_sensors = x.shape[:2]
        x = x.reshape(-1, *x.shape[-3:]).transpose((0, 3, 1, 2))  # to NCHW

        # inference
        logprob = self.drivable_model.run(None, {"input": x})[0]

        label = np.argmax(logprob, 1)
        drivable = np.concatenate([[label == i] for i in range(logprob.shape[1])], axis=1).astype(np.uint8) * 255

        if self.options["human_readable"]:
            num_sensors += 1
            drivable = np.concatenate([drivable, ((x + 1) / 2 * 255).astype(np.uint8)], axis=0)

        return drivable.reshape(num_cars, num_sensors, *drivable.shape[1:])
