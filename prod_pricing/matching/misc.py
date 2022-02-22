import lap
import numpy as np
import pandas as pd


def cvat_ann_dict2df(anndict: dict, labels=['price', 'pricetag']):
    dfs = []
    for im in anndict['annotations']['image']:
        imname = im['@name']
        boxes = im.get('box')
        if boxes:
            df = pd.DataFrame(boxes)
            df['image'] = imname
            dfs.append(df)
    df = pd.concat(dfs)
    df.columns = [x.replace('@', '') for x in df.columns]
    df = df.reset_index(drop=True)
    df['price'] = df.attribute.apply(lambda x: x.get('#text', '')
                                     if isinstance(x, dict) else None)
    df = df[df.label.isin(labels)].reset_index(drop=True)
    df[['xtl', 'ytl', 'xbr', 'ybr']] = df[['xtl', 'ytl', 'xbr',
                                           'ybr']].values.astype(float)
    df['box'] = df[['xtl', 'ytl', 'xbr', 'ybr']].values.tolist()
    df['width'] = df['xbr'] - df['xtl']
    df['height'] = df['ybr'] - df['ytl']
    df[['width', 'height']] = df[['width', 'height']].astype(float)
    df['area'] = df['height'] * df['width']
    df['xc'] = (df['xtl'] + df['xbr']) / 2
    df['yc'] = (df['ytl'] + df['ybr']) / 2
    df['area'] = df['area'].apply(lambda x: round(x, 2))
    df['xc'] = df['xc'].apply(lambda x: round(x, 2))
    df['yc'] = df['yc'].apply(lambda x: round(x, 2))
    df['center'] = df[['xc', 'yc']].values.tolist()
    return df


def cvat_ann_dict_poly2df(anndict: dict):
    poly_dfs = []
    for im in anndict['annotations']['image']:
        imname = im['@name']
        polys = im.get('polyline')
        if polys:
            if isinstance(polys, dict) == 1:
                polys = [polys]
            p_df = pd.DataFrame(polys)
            p_df['image'] = imname
            poly_dfs.append(p_df)
    pdf = pd.concat(poly_dfs)
    pdf.columns = [x.replace('@', '') for x in pdf.columns]
    pdf = pdf.reset_index(drop=True)
    pdf[['x1'
         ]] = pdf.points.apply(lambda x: float(x.split(';')[0].split(',')[0]))
    pdf[['y1'
         ]] = pdf.points.apply(lambda x: float(x.split(';')[0].split(',')[1]))
    pdf[['x2'
         ]] = pdf.points.apply(lambda x: float(x.split(';')[1].split(',')[0]))
    pdf[['y2'
         ]] = pdf.points.apply(lambda x: float(x.split(';')[1].split(',')[1]))
    pdf = pdf[['label', 'image', 'x1', 'y1', 'x2', 'y2']]
    return pdf


def modify_res_df(df):
    df = df.rename({'img_name': 'image'}, axis=1)
    df['box'] = df[['x1', 'y1', 'x2', 'y2']].values.tolist()
    df['xc'] = round((df['x1'] + df['x2']) / 2, 2)
    df['yc'] = round((df['y1'] + df['y2']) / 2, 2)
    df['center'] = df[['xc', 'yc']].values.tolist()
    return df


def bbox_overlaps(bboxes1,
                  bboxes2,
                  mode='iou',
                  eps=1e-6,
                  allow_exchange=False):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)
    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if allow_exchange:
        if bboxes1.shape[0] > bboxes2.shape[0]:
            bboxes1, bboxes2 = bboxes2, bboxes1
            ious = np.zeros((cols, rows), dtype=np.float32)
            exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(
            y_end - y_start, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def iofs(atlbrs, btlbrs):
    """Compute cost based on IoU.

    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    iofs = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if iofs.size == 0:
        return iofs

    iofs = bbox_overlaps(np.ascontiguousarray(atlbrs, dtype=np.float),
                         np.ascontiguousarray(btlbrs, dtype=np.float),
                         mode='iof')

    return iofs


def ious(atlbrs, btlbrs):
    """Compute cost based on IoU.

    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    iofs = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if iofs.size == 0:
        return iofs

    ious = bbox_overlaps(np.ascontiguousarray(atlbrs, dtype=np.float),
                         np.ascontiguousarray(btlbrs, dtype=np.float),
                         mode='iou')

    return ious


def linear_assignment(cost_matrix, thresh):
    """
    :param cost_matrix:
    :param thresh:
    :return:
    """
    if cost_matrix.size == 0:
        t1 = tuple(range(cost_matrix.shape[0]))
        t2 = tuple(range(cost_matrix.shape[1]))
        return np.empty((0, 2), dtype=int), t1, t2

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b
