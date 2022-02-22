import os
import pickle

import hydra
import imagesize
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.preprocessing import minmax_scale
from torch_geometric.data import Data
from tqdm import tqdm

from outputs.gnn_model import GMatcher


class BipartiteData(Data):
    def __init__(self, plen, edge_index, x_prod_geom, x_prod_visual, x_price):
        super(BipartiteData, self).__init__()
        self.edge_index = edge_index
        self.x_prod_geom = x_prod_geom
        self.x_prod_visual = x_prod_visual
        self.x_price = x_price
        self.num_nodes = len(x_prod_geom) + len(x_price)


@hydra.main(config_path='./', config_name='config')
def main(cfg):
    """Pre-cluster products and match by nearest distance."""
    model = GMatcher(6, 2048, 3, 64, 64)
    model.load_state_dict(torch.load(cfg.gnn_model_path))
    model.eval().cuda()

    # read product data
    init_prod_data = pd.read_csv(cfg.products_file)
    prod_data = init_prod_data.rename({'img_name': 'image'}, axis=1)
    img_names = list(prod_data.image.unique())

    # for feature normalization
    h_s = []
    w_s = []
    for im in tqdm(prod_data.image):
        img_w, img_h = imagesize.get(os.path.join(cfg.path2images, im))
        h_s.append(img_h)
        w_s.append(img_w)

    # product features
    prod_data['img_h'] = h_s
    prod_data['img_w'] = w_s
    prod_data['product_height'] = prod_data.y2 - prod_data.y1
    prod_data['product_width'] = prod_data.x2 - prod_data.x1
    prod_data['product_center_y'] = (prod_data.y2 + prod_data.y1) / 2
    prod_data['product_center_x'] = (prod_data.x2 + prod_data.x1) / 2
    prod_data['prod_feat_imnorm_height'] = prod_data[
        'product_height'] / prod_data['img_h']
    prod_data['prod_feat_imnorm_width'] = prod_data[
        'product_width'] / prod_data['img_w']
    prod_data['prod_feat_relnorm_height'] = prod_data[
        'product_height'] / prod_data.groupby(
            prod_data['image']).product_height.transform('mean')
    prod_data['prod_feat_relnorm_width'] = prod_data[
        'product_width'] / prod_data.groupby(
            prod_data['image']).product_width.transform('mean')
    prod_data[
        'prod_feat_y'] = prod_data['product_center_y'] / prod_data['img_h']
    prod_data[
        'prod_feat_x'] = prod_data['product_center_x'] / prod_data['img_w']

    # read price results
    price_data = pd.read_csv(cfg.result_file)
    price_data = price_data.dropna().reset_index(drop=True)
    price_data['image'] = price_data['imgpath'].apply(
        lambda x: x.split('/')[-1])
    price_data = price_data.merge(prod_data[['image', 'img_h',
                                             'img_w']].drop_duplicates())
    # price features
    price_data['price_box'] = price_data['price_box'].apply(eval)
    price_data['price_center_y'] = price_data['price_box'].apply(
        lambda x: (x[3] + x[1]) / 2)
    price_data['price_center_x'] = price_data['price_box'].apply(
        lambda x: (x[2] + x[0]) / 2)
    price_data[
        'price_feat_y'] = price_data['price_center_y'] / price_data['img_h']
    price_data[
        'price_feat_x'] = price_data['price_center_x'] / price_data['img_w']

    price_data['price_feat_price'] = price_data.groupby(
        price_data['image']).price.transform(lambda x: minmax_scale(x))

    prod_features = [x for x in prod_data.columns if 'prod_feat' in x]
    price_features = [x for x in price_data.columns if 'price_feat' in x]

    with open(cfg.prod_emb_file, 'rb') as f:
        embs = pickle.load(f)

    # FILTER!
    prod_data['center'] = prod_data[['product_center_x',
                                     'product_center_y']].values.tolist()
    price_data['center'] = price_data[['price_center_x',
                                       'price_center_y']].values.tolist()

    tr_data_list = []
    for im in img_names:
        # take image samples
        imdf_product = prod_data[prod_data.image ==
                                 im]  # .reset_index(drop=True)
        imdf_ptag = price_data[price_data.image == im]

        imdf_product['price'] = 0.00
        imdf_product['confidence'] = 0.01

        product_centers = [[float(o) for o in x]
                           for x in imdf_product['center'].values]
        ptag_centers = [[float(o) for o in x]
                        for x in imdf_ptag['center'].values]

        if not product_centers or not ptag_centers:
            tr_data_list.append(imdf_product)
            continue

        center_dists = cdist(product_centers, ptag_centers)
        # Do not consider pricelabels upper then a product
        y_dist_matrix = np.subtract.outer(imdf_product.product_center_y.values,
                                          imdf_ptag.price_center_y.values)
        mask = y_dist_matrix < 0
        center_dists[mask] = 1000

        imdf_ptag.loc[imdf_ptag.price == '', 'price'] = '0.00'
        gr_embgs = np.take(embs, imdf_product.index.tolist(), axis=0)
        # to torch
        gr_embgs = torch.tensor(gr_embgs, dtype=torch.float32)
        prod_geom_features = torch.tensor(imdf_product[prod_features].values,
                                          dtype=torch.float32)
        price_geom_features = torch.tensor(imdf_ptag[price_features].values,
                                           dtype=torch.float32)
        num_products = len(prod_geom_features)
        num_prices = len(price_geom_features)
        print(im, num_products, num_prices)
        edge_index = torch.cartesian_prod(
            torch.arange(num_products),
            torch.arange(num_products, num_products + num_prices)).T
        # to torch geometric data
        data = BipartiteData(len(prod_geom_features), edge_index,
                             prod_geom_features, gr_embgs, price_geom_features)
        data.cuda()
        with torch.no_grad():
            try:
                z = model.encode(data.x_prod_geom, data.x_prod_visual,
                                 data.x_price, data.edge_index)
                link_logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
                link_probs = link_logits.sigmoid()
                M = torch.zeros(num_products, num_prices)
                for v, (i, j) in enumerate(zip(edge_index[0], edge_index[1])):
                    M[i, j - num_products] = link_probs[v]
                vals, indices = M.max(axis=1)
                imdf_product['price'] = imdf_ptag.iloc[indices].price.values
                imdf_product['confidence'] = imdf_ptag.iloc[
                    indices].pricelabel_score.values * imdf_ptag.iloc[
                        indices].price_score.values
            except Exception as e:
                print(f'Proizoshla kakaya-to huinya: {e}')
        tr_data_list.append(imdf_product)
    tr_data = pd.concat(tr_data_list).reset_index(drop=True)
    init_prod_data['image'] = init_prod_data['img_name']
    predictions = init_prod_data[['image', 'x1', 'y1', 'x2', 'y2']].merge(
        tr_data[['image', 'x1', 'y1', 'x2', 'y2', 'price', 'confidence']])
    predictions.to_csv(cfg.predictions_path)


if __name__ == '__main__':
    main()
