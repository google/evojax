python3 train_masking.py --cnn-epochs 20 --algo OpenES --test-interval 100 --log-interval 50 --max-iter 1000 \
--max-steps 100 --mask-threshold 0.5 --val-fraction 0.4
python3 train_masking.py --cnn-epochs 20 --max-iter 0 --val-fraction 0.4
python3 train_masking.py --cnn-epochs 20 --algo OpenES --test-interval 100 --log-interval 50 --max-iter 1000 \
--max-steps 100 --mask-threshold 0.5 --val-fraction 0.6
python3 train_masking.py --cnn-epochs 20 --max-iter 0 --val-fraction 0.6
python3 train_masking.py --cnn-epochs 20 --algo OpenES --test-interval 100 --log-interval 50 --max-iter 1000 \
--max-steps 100 --mask-threshold 0.5 --val-fraction 0.8
python3 train_masking.py --cnn-epochs 20 --max-iter 0 --val-fraction 0.8