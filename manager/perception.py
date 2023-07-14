import json
import os

import numpy as np
import onnxruntime

import cv2


class PerceptionManager:
    def __init__(self,
                 global_options: dict,
                 gpu_index: int):
        self.options = {
            # "drivable_model": os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        #    "../static_files/models/ep_44_iou_0.9343.checkpoint.onnx"),
            "target_size": (160, 80),
            "max_velocity": 6,
            "human_readable": False
        }
        self.options.update(global_options.get("perception", {}))

        print("Perception Manager Options: ", json.dumps(self.options))

        # load models
        self.drivable_model = None
        if "drivable_model" in self.options:
            self.drivable_model = onnxruntime.InferenceSession(self.options["drivable_model"])

    def infer(self, images, bev_images, bev_seg_images, velocity):
        if self.drivable_model is not None:
            # original rgb for human readable
            orig_rgb = None
            if self.options["human_readable"]:
                orig_rgb = np.array(images).reshape(-1, *images[0][0].shape).transpose((0, 3, 1, 2))
                # orig_bev_rgb = np.array(bev_images).reshape(-1, *bev_images[0][0].shape).transpose((0, 3, 1, 2))

            # imagenet normalization
            dtype = np.float32
            x = np.array(images, dtype=dtype) / 255
            x = (x - np.array([0.485, 0.456, 0.406], dtype=dtype)) / np.array([0.229, 0.224, 0.225], dtype=dtype)
            # bev_x = np.array(bev_images, dtype=dtype) / 255
            # bev_x = (bev_x - np.array([0.485, 0.456, 0.406], dtype=dtype)) / np.array([0.229, 0.224, 0.225], dtype=dtype)

            # split batch
            num_cars, num_sensors = x.shape[:2]
            x = x.reshape(-1, *x.shape[-3:]).transpose((0, 3, 1, 2))  # to NCHW

            # inference
            logprob = self.drivable_model.run(None, {"input": x})[0]
            label = np.argmax(logprob, 1)
            # get drivable (ignore 0: non drivable)
            drivable = np.concatenate([[label == i] for i in range(1, logprob.shape[1])], axis=1).astype(np.uint8) * 255

            # concat velocity channel
            velocity_scalar = []
            for item in velocity:
                v = np.sqrt(item.x ** 2 + item.y ** 2) / self.options["max_velocity"]
                v = np.clip(v * 255, 0, 255).astype(np.uint8)
                velocity_scalar.append(v)
            drivable = np.concatenate([np.broadcast_to(velocity_scalar, (drivable.shape[0], 1, *drivable.shape[2:])), drivable], axis=1)

            if self.options["human_readable"]:
                num_sensors += 1
                drivable = np.concatenate([drivable, orig_rgb])

            # resize
            target_size = self.options["target_size"]
            if target_size is not None:
                drivable = np.array([
                    cv2.resize(img.transpose((1, 2, 0)), target_size, interpolation=cv2.INTER_NEAREST)
                    for img in drivable]).transpose((0, 3, 1, 2))

            return drivable.reshape((num_cars, num_sensors, *drivable.shape[1:]))
        else:
            target_size = self.options["target_size"]
            img_resized = np.array([
                cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                for img in images[0]]).transpose((0, 3, 1, 2))
            bev_img_resized = np.array([
                cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                for img in bev_images[0]]).transpose((0, 3, 1, 2))
            bev_seg_img_resized = np.array([
                cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                for img in bev_seg_images[0]]).transpose((0, 3, 1, 2))
            # print(bev_seg_images)
            # print(bev_seg_images.shape)
            # bev_seg_img_resized = np.array(bev_seg_images).transpose((0, 3, 1, 2))
            # print(bev_img_resized.shape)
            # print(bev_seg_img_resized.shape)
            # print(bev_img_resized)
            # print(bev_seg_img_resized)
            # cv2.imwrite(os.path.join("/figures", "resized_seg_bev_rgb.jpg"), cv2.cvtColor(bev_seg_img_resized[0].transpose((1, 2, 0)), cv2.COLOR_RGB2BGR))
            
            # cv2.imwrite(os.path.join("/figures", "expand_resized_seg_bev_rgb.jpg"), cv2.cvtColor(np.expand_dims(bev_seg_img_resized, 0).transpose((1, 2, 0)), cv2.COLOR_RGB2BGR))

            return np.expand_dims(img_resized, 0), np.expand_dims(bev_img_resized, 0), np.expand_dims(bev_seg_img_resized, 0)
