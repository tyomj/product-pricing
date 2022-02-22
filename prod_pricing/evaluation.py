import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@hydra.main(config_path='./', config_name='config')
def main(cfg):
    results_df = load_data(cfg.annotations_path, cfg.predictions_path)
    average_precision = evaluate_results(results_df)

    print('average precision = {}'.format(average_precision))


def load_data(annotations_path, results_path):
    annotations_df = pd.read_csv(annotations_path)
    results_df = pd.read_csv(results_path)

    if len(annotations_df) != len(results_df):
        print('Warning: each row in the given annotations'
              'file should have a corresponding row'
              'with prediction in the results file')

    results_df['groundtruth'] = annotations_df['price']
    return results_df


def evaluate_results(results_df, plot=False):
    sorted_df = results_df.sort_values('confidence', ascending=False)

    tp = np.array(sorted_df['price'] == sorted_df['groundtruth']).astype(float)
    p = np.ones(len(sorted_df))
    num_positives = len(sorted_df)

    tp_cumsum = np.cumsum(tp)
    p_cumsum = np.cumsum(p)

    precision = tp_cumsum / p_cumsum
    coverage = p_cumsum / num_positives
    average_precision = np.mean(precision)

    if plot:
        plt.plot(coverage, precision)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    return average_precision


if __name__ == '__main__':
    main()
