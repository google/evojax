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

datasets_tuple = full_data_loader()
study = optuna.create_study(direction="maximize",
                            study_name="mnist_baselines",
                            storage=f'sqlite:///{log_dir}/optuna_hparam_search.db',
                            )


for _ in range(10):
    trial = study.ask()

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    l1_reg_lambda = trial.suggest_float("l1_reg_lambda", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2**i for i in range(7, 11)])
    use_task_labels = trial.suggest_categorical("use_task_labels", [True, False])

    _, val_accuracy = run_mnist_training(logger,
                                         wandb_logging=False,
                                         seed=0,
                                         num_epochs=50,
                                         evo_epoch=0,
                                         learning_rate=learning_rate,
                                         cnn_batch_size=batch_size,
                                         state=None,
                                         mask_params=None,
                                         datasets_tuple=datasets_tuple,
                                         early_stopping=True,
                                         # These are the parameters for the other
                                         # sparsity baseline types
                                         use_task_labels=use_task_labels,
                                         l1_pruning_proportion=None,
                                         l1_reg_lambda=l1_reg_lambda,
                                         dropout_rate=None)
    study.tell(trial, val_accuracy)

trial = study.best_trial
logger.info(f'Best Validation Accuracy: {trial.value:.4}')
logger.info(f'Best Params:')
for key, value in trial.params.items():
    logger.info(f'-> {key}: {value}')

df = study.trials_dataframe()
df.to_csv(f'{os.path.join(log_dir, "dataframes",time.strftime("%m%d_%H%M"))}.csv')
