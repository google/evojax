import configparser
import os
import time
import optuna
from evojax.train_masking import run_train_masking
from evojax.datasets import full_data_loader
from evojax.util import create_logger

log_dir = './log/optuna'
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
logger = create_logger(name='SWEEP', log_dir=log_dir, debug=False)

seed = 0
datasets_tuple = full_data_loader()
study = optuna.create_study(direction="maximize",
                            study_name=f"mnist_evo_seed_{seed}",
                            storage=f'sqlite:///{log_dir}/optuna_hparam_search.db',
                            )

params_dict = dict(
    algo=None,
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
    cnn_epochs=20,
    cnn_lr=1e-3,
)

for _ in range(6):
    trial = study.ask()

    test_params = dict(
        algo=trial.suggest_categorical("algo", ["PGPE", "OpenES"]),
        pop_size=trial.suggest_categorical("pop_size", [8, 16, 32]),
        mask_threshold=trial.suggest_float("mask_threshold", 0.3, 0.7),
        max_iter=trial.suggest_int("max_iter", 1, 4, log=True),
        evo_epochs=trial.suggest_int("evo_epochs", 1, 20, log=False),
        test_interval=trial.suggest_int("test_interval", 1, 20, log=False),
        center_lr=trial.suggest_float("center_lr", 0, 0.1),
        std_lr=trial.suggest_float("std_lr", 0, 0.2),
        init_std=trial.suggest_float("init_std", 0, 0.1),
    )
    params_dict.update(test_params)

    _, val_accuracy = run_train_masking(**params_dict, logger=logger, config_dict=params_dict)
    study.tell(trial, val_accuracy)

trial = study.best_trial
logger.info(f'Best Validation Accuracy: {trial.value:.4}')
logger.info(f'Best Params:')
for key, value in trial.params.items():
    logger.info(f'-> {key}: {value}')

df = study.trials_dataframe()
dataframe_dir = os.path.join(log_dir, "dataframes")
os.makedirs(dataframe_dir, exist_ok=True)
df.to_csv(f'{os.path.join(dataframe_dir, "evo")}_{time.strftime("%m%d_%H%M")}.csv')
