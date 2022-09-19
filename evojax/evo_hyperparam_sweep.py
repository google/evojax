import os
import time
import optuna
import argparse
from evojax.train_masking import run_train_masking
from evojax.datasets import full_data_loader
from evojax.util import create_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial-count', type=int, default=5, help='How many trials to run.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for training.')
    parser.add_argument('--dropout', action='store_true', help='Use dropout and masking.')
    parser.add_argument('--pixel-input', action='store_true', help='Use pixel input for the mask.')
    parser.add_argument('--test', action='store_true', help='Check test acc.')
    parsed_config, _ = parser.parse_known_args()
    return parsed_config


if __name__ == "__main__":
    config = parse_args()

    log_dir = './log/optuna'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(name='SWEEP', log_dir=log_dir, debug=False)

    seed = config.seed
    datasets_tuple = full_data_loader()
    study = optuna.create_study(direction="maximize",
                                study_name=f"pixel{'_test' if config.test else ''}_evo_seed_{seed}",
                                storage=f'sqlite:///{log_dir}/optuna_hparam_search.db',
                                load_if_exists=True)
    # study = optuna.create_study(direction="maximize",
    #                             study_name=f"dropout_test_evo_seed_0",
    #                             storage=f'sqlite:///{log_dir}/optuna_hparam_search.db',
    #                             load_if_exists=True)
    #
    params_dict = dict(
        algo=None,
        pixel_input=config.pixel_input,
        pop_size=8,
        batch_size=1024,
        mask_threshold=0.5,
        max_iter=100,
        max_steps=1,
        evo_epochs=1,
        test_interval=10,
        log_interval=10,
        center_lr=0.006,
        std_lr=0.089,
        init_std=0.039,
        cnn_epochs=5,
        cnn_lr=1e-3,
        log_evo=False,
        early_stopping=False,
        datasets_tuple=datasets_tuple
    )

    logger.info(f'Running {config.trial_count} Trials')
    for i in range(config.trial_count):
        logger.info(f'Starting Trial {i}/{config.trial_count}')

        trial = study.ask()

        test_params = dict(
            algo=trial.suggest_categorical("algo", ["PGPE", "OpenES"]),
            pop_size=trial.suggest_categorical("pop_size", [16, 32, 64, 128]),
            mask_threshold=trial.suggest_float("mask_threshold", 0.4, 0.6),
            max_iter=trial.suggest_int("max_iter", 20, 300, log=True),
            evo_epochs=trial.suggest_int("evo_epochs", 0, 10, log=False),
            cnn_epochs=trial.suggest_int("cnn_epochs", 1, 5, log=False),
            test_interval=trial.suggest_int("test_interval", 5, 20, log=False),
            center_lr=trial.suggest_float("center_lr", 0, 0.1),
            std_lr=trial.suggest_float("std_lr", 0, 0.2),
            init_std=trial.suggest_float("init_std", 0, 0.2),
            dropout_rate=trial.suggest_float("dropout", 0.3, 0.7) if config.dropout else None,
        )
        params_dict.update(test_params)

        accuracy_dict = run_train_masking(**params_dict, logger=logger, config_dict=params_dict)
        if config.test:
            opt_value = accuracy_dict['test'][-1]
        else:
            opt_value = accuracy_dict['validation'][-1]

        study.tell(trial, opt_value)

    trial = study.best_trial
    logger.info(f'Best Validation Accuracy: {trial.value:.4}')
    logger.info(f'Best Params:')
    for key, value in trial.params.items():
        logger.info(f'-> {key}: {value}')

    df = study.trials_dataframe()
    dataframe_dir = os.path.join(log_dir, "dataframes")
    os.makedirs(dataframe_dir, exist_ok=True)
    df.to_csv(f'{os.path.join(dataframe_dir, "evo")}_{time.strftime("%m%d_%H%M")}.csv')
