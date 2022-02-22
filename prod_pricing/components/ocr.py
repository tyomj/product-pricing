import logging
import os
import sys

import numpy as np
import paddle
import tools.program as program
from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import init_model

sys.path.append('/product-pricing/PaddleOCR')

os.environ['FLAGS_allocator_strategy'] = 'auto_growth'

logger = logging.getLogger()


class PPOCRPredictor:
    def __init__(self, config_path, device):
        self.config_path = config_path
        self.config = program.load_config(config_path)
        self.global_config = self.config['Global']
        self.device = device

        self.postprocess = build_post_process(self.config['PostProcess'],
                                              self.global_config)
        if hasattr(self.postprocess, 'character'):
            self.config['Architecture']['Head']['out_channels'] = len(
                getattr(self.postprocess, 'character'))

        self.model = build_model(self.config['Architecture'])
        init_model(self.config, self.model, logger)
        self.model.eval()

        self._get_transorfms()

    def _get_transorfms(self):
        # create data ops
        transforms = []
        for op in self.config['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                continue
            elif op_name in ['RecResizeImg']:
                op[op_name]['infer_mode'] = True
            elif op_name == 'KeepKeys':
                op[op_name]['keep_keys'] = ['image']
            transforms.append(op)
        self.ops = create_operators(transforms, self.global_config)

    def predict(self, img):
        data = {'image': img}
        batch = transform(data, self.ops)
        images = np.expand_dims(batch[0], axis=0)
        images = paddle.to_tensor(images)
        preds = self.model(images)
        post_result = self.postprocess(preds)[0]
        return post_result
