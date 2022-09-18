import os
import time
import optuna
import argparse

from evojax.mnist_cnn import run_mnist_training
from evojax.datasets import full_data_loader
from evojax.util import create_logger


def parse_args():
    parser = argparse.ArgumentParser()
    # Params for the CNN
    parser.add_argument('--use-task-labels', action='store_true', help='Input the task labels to the CNN.')
    parser.add_argument('--l1-reg', action='store_true', help='Test L1 regularisation parameters.')
    parser.add_argument('--l1-pruning', action='store_true', help='Test L1 pruning parameters.')
    parser.add_argument('--dropout', action='store_true', help='Test dropout rate.')

    # General params
    parser.add_argument('--seed', type=int, default=0, help='Random seed for training.')

    parsed_config, _ = parser.parse_known_args()
    return parsed_config


if __name__ == "__main__":
    config = parse_args()

    log_dir = './log/optuna'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(name='SWEEP', log_dir=log_dir, debug=False)

    seed = 0
    datasets_tuple = full_data_loader()
    study = optuna.create_study(direction="maximize",
                                study_name=f"mnist_baselines_seed_{seed}",
                                storage=f'sqlite:///{log_dir}/optuna_hparam_search.db',
                                load_if_exists=True)

    base_config = dict(logger=logger,
                       wandb_logging=False,
                       seed=seed,
                       num_epochs=50,
                       learning_rate=1e-3,
                       cnn_batch_size=1024,
                       datasets_tuple=datasets_tuple,
                       early_stopping=True,
                       # These are the parameters for the other
                       # sparsity baseline types
                       use_task_labels=False,
                       l1_pruning_proportion=None,
                       l1_reg_lambda=None,
                       dropout_rate=None)

    trial_states = {}

    for _ in range(3):
        trial = study.ask()

        if config.use_task_labels:
            use_task_labels = trial.suggest_categorical("use_task_labels", [True, False])
            base_config.update({"use_task_labels": use_task_labels})
        if config.l1_reg:
            l1_reg_lambda = trial.suggest_float("l1_reg_lambda", 1e-5, 1e-3, log=True)
            base_config.update({"l1_reg_lambda": l1_reg_lambda})
        if config.l1_pruning:
            l1_pruning_proportion = trial.suggest_float("l1_pruning_proportion", 0.05, 0.8, step=0.05, log=False)
            base_config.update({"l1_pruning_proportion": l1_pruning_proportion})
        if config.dropout:
            dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.5, step=0.05, log=False)
            base_config.update({"dropout_rate": dropout_rate})

        cnn_state, val_accuracy = run_mnist_training(**base_config)
        trial_states[trial.number] = cnn_state
        study.tell(trial, val_accuracy)

    trial = study.best_trial
    logger.info(f'Best Validation Accuracy: {trial.value:.4}')
    logger.info(f'Best Params:')
    for key, value in trial.params.items():
        logger.info(f'-> {key}: {value}')

    # base_config.update(trial.params)
    # base_config["state"] = trial_states[trial.number]
    # _, best_params_test_acc = run_mnist_training(eval_only=True, **base_config)
    # logger.info(f'Corresponding Best Test Accuracy: {best_params_test_acc:.4}')

    df = study.trials_dataframe()
    dataframe_dir = os.path.join(log_dir, "dataframes")
    os.makedirs(dataframe_dir, exist_ok=True)
    df.to_csv(f'{os.path.join(dataframe_dir, "baselines")}_{time.strftime("%m%d_%H%M")}.csv')
