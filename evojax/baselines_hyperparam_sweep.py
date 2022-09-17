import os
import time
import optuna
from evojax.mnist_cnn import run_mnist_training
from evojax.datasets import full_data_loader
from evojax.util import create_logger

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


for _ in range(3):
    trial = study.ask()

    # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    l1_reg_lambda = trial.suggest_float("l1_reg_lambda", 1e-5, 1e-3, log=True)
    # batch_size = trial.suggest_categorical("batch_size", [2**i for i in range(7, 11)])
    # use_task_labels = trial.suggest_categorical("use_task_labels", [True, False])
    base_config.update({"l1_reg_lambda": l1_reg_lambda})

    _, val_accuracy = run_mnist_training(**base_config)
    study.tell(trial, val_accuracy)

trial = study.best_trial
logger.info(f'Best Validation Accuracy: {trial.value:.4}')
logger.info(f'Best Params:')
for key, value in trial.params.items():
    logger.info(f'-> {key}: {value}')

base_config.update(trial.params)
_, best_params_test_acc = run_mnist_training(eval_only=True, **base_config)
logger.info(f'Corresponding Best Test Accuracy: {best_params_test_acc:.4}')

df = study.trials_dataframe()
dataframe_dir = os.path.join(log_dir, "dataframes")
os.makedirs(dataframe_dir, exist_ok=True)
df.to_csv(f'{os.path.join(dataframe_dir, time.strftime("%m%d_%H%M"))}.csv')
