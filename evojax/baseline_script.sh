#!/bin/bash
python3 train_masking.py --cnn-epochs 30
python3 train_masking.py --cnn-epochs 30 --cnn-labels

python3 train_masking.py --evo-epochs 5 --cnn-epochs 6 --algo PGPE --max-iter 100 --test-interval 10000 --log-interval 10000
python3 train_masking.py --evo-epochs 10 --cnn-epochs 3 --algo PGPE --max-iter 100 --test-interval 10000 --log-interval 10000
python3 train_masking.py --evo-epochs 15 --cnn-epochs 2 --algo PGPE --max-iter 100 --test-interval 10000 --log-interval 10000

python3 train_masking.py --evo-epochs 10 --cnn-epochs 3 --algo PGPE --max-iter 1000 --test-interval 10000 --log-interval 10000
python3 train_masking.py --evo-epochs 15 --cnn-epochs 2 --algo PGPE --max-iter 1000 --test-interval 10000 --log-interval 10000
python3 train_masking.py --evo-epochs 30 --cnn-epochs 1 --algo PGPE --max-iter 1000 --test-interval 10000 --log-interval 10000

python3 train_masking.py --evo-epochs 3 --cnn-epochs 10 --algo PGPE --max-iter 2000 --test-interval 10000 --log-interval 10000
python3 train_masking.py --evo-epochs 6 --cnn-epochs 5 --algo PGPE --max-iter 2000 --test-interval 10000 --log-interval 10000
python3 train_masking.py --evo-epochs 12 --cnn-epochs 1 --algo PGPE --max-iter 2000 --test-interval 10000 --log-interval 10000

python3 train_masking.py --evo-epochs 50 --cnn-epochs 1 --algo PGPE --max-iter 10 --test-interval 10000 --log-interval 10000