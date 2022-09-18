import os
import time
import argparse
import pandas as pd
from evojax.train_masking import run_train_masking
from google.cloud import storage
from evojax.datasets import full_data_loader
from evojax.util import create_logger


if __name__ == "__main__":

    log_dir = './log/results'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(name='RESULTS', log_dir=log_dir, debug=False)

    client = storage.Client()
    bucket = client.get_bucket("evojax-bucket")
    file_name = f'results_run_{time.strftime("%m%d_%H%M")}.csv'


    number_of_seeds = 5
    datasets_tuple = full_data_loader()

    baseline_dict = dict(
        algo=None,
        pop_size=16,
        batch_size=1024,
        cnn_epochs=20,
        evo_epochs=0,
        cnn_lr=1e-3,
        datasets_tuple=datasets_tuple
    )

    baseline_results = {}
    for s in range(number_of_seeds):
        baseline_results[s] = run_train_masking(**baseline_dict, logger=logger, config_dict=baseline_dict)

    task_labels_dict = dict(**baseline_dict, use_task_labels=True)
    task_labels_results = {}
    for s in range(number_of_seeds):
        task_labels_results[s] = run_train_masking(**task_labels_dict, logger=logger, config_dict=task_labels_dict)

    dropout_dict = dict(**baseline_dict, dropout_rate=0.5)
    dropout_results = {}
    for s in range(number_of_seeds):
        dropout_results[s] = run_train_masking(**dropout_dict, logger=logger, config_dict=dropout_dict)

    l1_reg_dict = dict(**baseline_dict, l1_reg_rate=3e-5)
    l1_reg_results = {}
    for s in range(number_of_seeds):
        l1_reg_results[s] = run_train_masking(**l1_reg_dict, logger=logger, config_dict=l1_reg_dict)

    l1_pruning_dict = dict(**baseline_dict, l1_pruning_proportion=1e-5)
    l1_pruning_results = {}
    for s in range(number_of_seeds):
        l1_pruning_results[s] = run_train_masking(**l1_pruning_dict, logger=logger, config_dict=l1_pruning_dict)


    blob = bucket.blob(file_name)
    blob.upload_from_filename(file_name)






