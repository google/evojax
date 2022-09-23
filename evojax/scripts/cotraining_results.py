import os
import time
import argparse
import pandas as pd
from itertools import combinations, chain
from evojax.train_masking import run_train_masking
from google.cloud import storage
from evojax.datasets import full_data_loader, DATASET_LABELS
from evojax.util import create_logger


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--number-of-seeds', type=int, default=5, help='How many seeds to test.')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs.')

    parsed_config, _ = parser.parse_known_args()
    return parsed_config


def run_and_format_results(number_of_seeds, config_dict, run_name):
    results = {}
    for s in range(number_of_seeds):
        results[s] = run_train_masking(**config_dict, config_dict=config_dict, seed=s)
    return {(run_name, k2, k1): v2 for k1, v1 in results.items() for k2, v2 in v1.items()}


if __name__ == "__main__":
    config = parse_args()

    log_dir = './log/results'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(name='RESULTS', log_dir=log_dir, debug=False)

    client = storage.Client()
    bucket = client.get_bucket("evojax-bucket")
    file_name = f'dataset_comparison_{time.strftime("%m%d_%H%M")}.csv'
    file_path = os.path.join(log_dir, file_name)

    datasets = list(DATASET_LABELS.keys())
    dataset_choices = chain([combinations(datasets, 1), combinations(datasets, 2)])

    full_results = {}
    for ds in dataset_choices:
        datasets_tuple = full_data_loader(dataset_names=ds)

        baseline_dict = dict(
            batch_size=1024,
            cnn_epochs=config.epochs,
            max_iter=0,
            cnn_lr=1e-3,
            early_stopping=True,
            datasets_tuple=datasets_tuple,
            logger=logger
        )

        baseline_results = run_and_format_results(config.number_of_seeds, baseline_dict, 'baseline')
        full_results.update(baseline_results)

    try:
        series_dict = {k: pd.Series(v) for k, v in full_results.items()}

        df = pd.DataFrame.from_dict(series_dict, orient='index')
        df.to_csv(file_path)

        logger.info(f'Saving results as: {file_name}')

        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_path)
    except:
        import ipdb
        ipdb.set_trace()

