import pickle

import hydra
import numpy as np
import pandas as pd
from matching.misc import modify_res_df
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist


def merge_the_same_products(prod_df, path2embs):
    with open(path2embs, 'rb') as f:
        embs = pickle.load(f)

    for name, group in prod_df.groupby('image'):
        if len(group) >= 2:
            gr_embgs = np.take(embs, group.index.tolist(), axis=0)
            cosine_mat = np.round(
                cdist(gr_embgs, gr_embgs, metric='cosine') +
                np.eye(len(gr_embgs)), 2)
            x_mat = np.round(
                cdist(group['xc'].values.reshape(-1, 1),
                      group['xc'].values.reshape(-1, 1)) +
                np.eye(len(gr_embgs)), 2)
            y_mat = np.round(
                cdist(group['yc'].values.reshape(-1, 1),
                      group['yc'].values.reshape(-1, 1)) +
                np.eye(len(gr_embgs)), 2)
            some_mat = (cosine_mat < 0.20) * (x_mat < 250) * (y_mat < 30)
            _, labels = connected_components(some_mat.astype(int))
            if len(set(labels)) < len(gr_embgs):
                gr = pd.DataFrame([group.xc.values, group.yc.values, labels],
                                  index=['xc', 'yc', 'label']).T
                xc_remapper = gr.groupby('label').xc.min().to_dict()
                yc_remapper = gr.groupby('label').yc.min().to_dict()

                xcs = [xc_remapper[float(x)] for x in labels]
                xys = [yc_remapper[float(x)] for x in labels]

                prod_df.loc[group.index, 'xc'] = xcs
                prod_df.loc[group.index, 'xy'] = xys

    return prod_df


@hydra.main(config_path='./', config_name='config')
def main(cfg):
    """Pre-cluster products and match by nearest distance."""
    df = pd.read_csv(cfg.result_file)
    df = df.dropna().reset_index(drop=True)
    df['image'] = df['imgpath'].apply(lambda x: x.split('/')[-1])
    df.price_box = df.price_box.apply(eval)
    df['center'] = df.price_box.apply(lambda x: [(x[2] + x[0]) / 2,
                                                 (x[3] + x[1]) / 2])
    df['xc'] = df.price_box.apply(lambda x: x[0])
    df['yc'] = df.price_box.apply(lambda x: x[1])

    init_prod_data = pd.read_csv(cfg.products_file)
    prod_data = modify_res_df(init_prod_data)
    prod_data = merge_the_same_products(prod_data, cfg.prod_emb_file)
    prod_data['center'] = prod_data[['xc', 'yc']].values.tolist()

    img_names = list(prod_data.image.unique())

    tr_data_list = []
    for im in img_names:
        imdf_product = prod_data[prod_data.image == im].reset_index(drop=True)
        imdf_ptag = df[df.image == im]
        imdf_ptag.loc[imdf_ptag.price == '', 'price'] = '0.00'

        product_centers = [[float(o) for o in x]
                           for x in imdf_product['center'].values]
        ptag_centers = [[float(o) for o in x]
                        for x in imdf_ptag['center'].values]
        prices = imdf_ptag.price.values
        imdf_product['price'] = 0.0
        imdf_product['confidence'] = 0.01

        if not product_centers or not ptag_centers:
            tr_data_list.append(imdf_product)
            continue

        center_dists = cdist(product_centers, ptag_centers)
        # Do not consider pricelabels upper then a product
        y_dist_matrix = np.subtract.outer(imdf_product.yc.values,
                                          imdf_ptag.yc.values)
        mask = y_dist_matrix >= 0
        center_dists[mask] = 3000

        for i in range(center_dists.shape[0]):
            closest_price = center_dists[i].argmin(axis=0)
            imdf_product.loc[i, 'price'] = prices[closest_price]
            imdf_product.loc[i, 'confidence'] = imdf_ptag.iloc[
                closest_price].pricelabel_score * imdf_ptag.iloc[
                    closest_price].price_score
        tr_data_list.append(imdf_product)
    tr_data = pd.concat(tr_data_list).reset_index(drop=True)
    init_prod_data['image'] = init_prod_data['img_name']
    predictions = init_prod_data[['image', 'x1', 'y1', 'x2', 'y2']].merge(
        tr_data[['image', 'x1', 'y1', 'x2', 'y2', 'price', 'confidence']])
    predictions.to_csv(cfg.predictions_path)


if __name__ == '__main__':
    main()
