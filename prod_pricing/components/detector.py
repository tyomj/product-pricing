import sys

import numpy
import torch
from datasets import letterbox
from general import non_max_suppression, scale_coords
from models.experimental import attempt_load

sys.path.append('/product-pricing/yolov5')
sys.path.append('/product-pricing/yolov5/utils')


class YoloV5Predictor:
    def __init__(self, weights_path, nms_conf, nms_iou, device):
        self.weights_path = weights_path
        self.nms_conf = nms_conf
        self.nms_iou = nms_iou
        self.device = device
        self.model = attempt_load(self.weights_path, map_location=self.device)

    def infer(self, img):
        return self.model(img)[0]

    def nms(self, pred):
        return non_max_suppression(pred, self.nms_conf, self.nms_iou)[-1]

    def preproces(self, img0):
        img = letterbox(img0, new_shape=640, auto=False)[0]
        # to RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = numpy.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img, img0.shape[:2]

    def postprocess(self, dets, shape, ori_shape):
        dets[:, :4] = scale_coords(img1_shape=shape,
                                   coords=dets[:, :4],
                                   img0_shape=ori_shape)
        dets = dets.cpu().numpy()
        bboxes = dets[:, :4]
        scores = dets[:, 4]
        return bboxes, scores

    def predict(self, img0):
        img, img0_shape = self.preproces(img0)
        preds = self.infer(img)
        nms_preds = self.nms(preds)
        results = self.postprocess(nms_preds, (640, 640), img0_shape)
        return results
