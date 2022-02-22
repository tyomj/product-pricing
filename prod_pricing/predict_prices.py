import sys
from glob import glob

import cv2
import hydra
import numpy as np
import pandas as pd
import torch
from components.detector import YoloV5Predictor
from components.ocr import PPOCRPredictor
from src.pl_modules import InferenceModel
from tqdm import tqdm

sys.path.append('/product-pricing/prc-seg')


class PricePredictor:
    def __init__(self, det_path, loc_path, rec_conf_path, nms_conf, nms_iou,
                 device, **kwargs):
        self.det = YoloV5Predictor(weights_path=det_path,
                                   nms_conf=nms_conf,
                                   nms_iou=nms_iou,
                                   device=device)
        self.loc = InferenceModel.load_from_checkpoint(loc_path).eval()
        self.rec = PPOCRPredictor(config_path=rec_conf_path, device=device)

    def predict_single(self, image_path):

        img0 = cv2.imread(image_path)

        # 1. Detection
        boxes, scores = self.det.predict(img0)

        # 2. Segmentation
        price_boxes = []
        price_images = []
        img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB).astype(np.float32)
        price_boxes = []
        price_images = []
        for box in boxes:
            try:
                x1, y1, x2, y2 = box.astype(int)
                pricelabel_img = img0_rgb[y1:y2, x1:x2, :]
                h, w, _ = pricelabel_img.shape
                with torch.no_grad():
                    mask = self.loc(pricelabel_img)[0][0].numpy()
                resized_masks = cv2.resize(mask, (w, h))
                px1, py1, px2, py2 = self.loc.mask2bbox(resized_masks)
                # to absolute coords
                price_box = [px1 + x1, py1 + y1, px2 + x1, py2 + y1]
                p_image = img0[price_box[1]:price_box[3],
                               price_box[0]:price_box[2], :]
                ph, pw, _ = p_image.shape
                assert ph * pw > 16 * 16
                price_boxes.append(price_box)
                price_images.append(p_image)
            except Exception as e:
                print(f'Proizoshla kakaya-to huinya: {e}')
                price_boxes.append(None)
                price_images.append(None)

        # 3. OCR
        prices = []
        price_scores = []
        for price_img in price_images:
            if price_img is None:
                prices.append(None)
                price_scores.append(None)
            else:
                price_value, ocr_score = self.rec.predict(price_img)
                prices.append(price_value)
                price_scores.append(ocr_score)

        return boxes.astype(
            int).tolist(), scores.tolist(), price_boxes, prices, price_scores


@hydra.main(config_path='./', config_name='config')
def main(cfg):
    dfs = []
    image_paths = glob(cfg.path2images + '**.jpg')
    pp = PricePredictor(**cfg)
    for impath in tqdm(image_paths):
        res = pp.predict_single(impath)
        df = pd.DataFrame(res).T
        df.columns = [
            'pricelabel_box', 'pricelabel_score', 'price_box', 'price',
            'price_score'
        ]
        df['imgpath'] = impath
        dfs.append(df)
        results = pd.concat(dfs).reset_index(drop=True)
        results.to_csv(cfg.result_file, index=False)


if __name__ == '__main__':
    main()
