import os
import optuna
from evojax.mnist_cnn import run_mnist_training
from evojax.datasets import full_data_loader
from evojax.util import create_logger

log_dir = './log/masking'
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
logger = create_logger(name='SWEEP', log_dir=log_dir, debug=False)

datasets_tuple = full_data_loader()
study = optuna.create_study(direction="maximize")


for _ in range(20):
    trial = study.ask()

    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2**i for i in range(7, 11)])

    _, val_accuracy = run_mnist_training(logger,
                                         wandb_logging=False,
                                         seed=0,
                                         num_epochs=50,
                                         evo_epoch=0,
                                         learning_rate=learning_rate_init,
                                         cnn_batch_size=batch_size,
                                         state=None,
                                         mask_params=None,
                                         datasets_tuple=datasets_tuple,
                                         early_stopping=True,
                                         # These are the parameters for the other
                                         # sparsity baseline types
                                         use_task_labels=False,
                                         l1_pruning_proportion=None,
                                         l1_reg_lambda=None,
                                         dropout_rate=None)
    study.tell(trial, val_accuracy)
