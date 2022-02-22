import hydra
import numpy as np
import pandas as pd
from matching.misc import modify_res_df
from scipy.spatial.distance import cdist


@hydra.main(config_path='./', config_name='config')
def main(cfg):
    """Match by nearest distance."""
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
