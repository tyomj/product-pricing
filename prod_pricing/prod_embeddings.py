import os
import pickle

import cv2
import hydra
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

INPUT_SIZE = 112


class ProductDataset(Dataset):
    def __init__(self, csv_path, path_to_images):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.path_to_images = path_to_images
        self.d = self.df.to_dict(orient='index')
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize(int(INPUT_SIZE / 0.875)),
            transforms.CenterCrop(int(INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.d)

    def __getitem__(self, index):
        row = self.d[index]
        impath = os.path.join(self.path_to_images, row['img_name'])
        full_image = cv2.imread(impath)
        crop = full_image[row['y1']:row['y2'], row['x1']:row['x2'], :]
        image = self.transform(crop)
        return image


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.num_classes = 50030
        self.backbone = torch.hub.load('zhanghang1989/ResNeSt',
                                       'resnest50',
                                       pretrained=False)
        self.backbone.fc = nn.Sequential()
        self.feature_len = 2048
        self.classifier = nn.Linear(self.feature_len,
                                    self.num_classes,
                                    bias=True)

    def load_model(self, model_path):
        pretrain_dict = torch.load(model_path, map_location='cuda')
        pretrain_dict = pretrain_dict[
            'state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith('module'):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print('Model has been loaded...')

    def forward(self, x, **kwargs):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        # x = self.classifier(x)
        return x


@hydra.main(config_path='./', config_name='config')
def main(cfg):

    # init model
    model = Network()
    model.load_model(cfg.emb_model_path)
    model.eval().to(cfg.device)

    # init dataset
    dataset = ProductDataset(csv_path=cfg.products_file,
                             path_to_images=cfg.path2images)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # infer
    embeddings = []
    for batch in tqdm(dataloader):
        batch = batch.to(cfg.device)
        with torch.no_grad():
            out = model(batch).detach().cpu()
        embeddings.append(out)
    embeddings = torch.cat(embeddings)
    embeddings = embeddings.numpy()

    with open(cfg.prod_emb_file, 'wb') as f:
        pickle.dump(embeddings, f)


if __name__ == '__main__':
    main()
