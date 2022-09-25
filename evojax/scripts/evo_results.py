import os
import time
import argparse
import pandas as pd
from evojax.train_masking import run_train_masking
from google.cloud import storage
from evojax.datasets import full_data_loader, DATASET_LABELS
from evojax.util import create_logger


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--number-of-seeds', type=int, default=5, help='How many seeds to test.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs.')
    parser.add_argument('--early-stopping-count', type=int, help='Number of epochs.')

    parser.add_argument('--dropout-rate', type=float, help='The rate for dropout layers in CNN.')
    parser.add_argument('--mask-threshold', type=float, help='The threshold for hard masking.')

    parsed_config, _ = parser.parse_known_args()
    return parsed_config


def run_and_format_results(config_dict, run_name):
    results = {}
    for s in range(number_of_seeds):
        print(f'\nStarting seed: {s}\n')
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
    file_name = f'evo{"_hard" if config.mask_threshold else "_soft"}' \
                f'{"_dropout" if config.dropout_rate else ""}' \
                f'{"_es" if config.early_stopping_count else ""}_{time.strftime("%m%d_%H%M")}.csv'
    file_path = os.path.join(log_dir, file_name)

    logger.info(f"Running results file: {file_name}")

    number_of_seeds = config.number_of_seeds
    datasets = list(DATASET_LABELS.keys())
    datasets_tuple = full_data_loader(dataset_names=datasets)

    baseline_dict = dict(dataset_names=datasets,
                         batch_size=1024,
                         cnn_epochs=config.epochs,
                         cnn_lr=1e-3,
                         dropout_rate=config.dropout_rate,
                         early_stopping_count=config.early_stopping_count,
                         datasets_tuple=datasets_tuple,
                         logger=logger
                         )
    full_results = {}

    masking_params = dict(algo="OpenES",
                          pop_size=8,
                          mask_threshold=config.mask_threshold,
                          max_iter=720,
                          max_steps=50,
                          test_interval=160,
                          log_interval=100,
                          center_lr=0.0825,
                          std_lr=0.08,
                          init_std=0.045)

    masking_dict = dict(**baseline_dict, **masking_params)
    masking_results = run_and_format_results(masking_dict, f'masking{"_hard" if config.mask_threshold else "_soft"}'
                                                           f'{"_dropout" if config.dropout_rate else ""}')
    full_results.update(masking_results)

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

