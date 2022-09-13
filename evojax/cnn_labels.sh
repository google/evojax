#!/bin/bash
python3 train_masking.py --cnn-epochs 30 --cnn-labels --seed 0
python3 train_masking.py --cnn-epochs 30 --cnn-labels --seed 1
python3 train_masking.py --cnn-epochs 30 --cnn-labels --seed 2
python3 train_masking.py --cnn-epochs 30 --cnn-labels --seed 3
python3 train_masking.py --cnn-epochs 30 --cnn-labels --seed 4
