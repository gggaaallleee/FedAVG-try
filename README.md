python main_nn.py --model mlp --dataset mnist --epochs 50 --gpu 0
python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0 --all_clients


python main_fed.py --dataset mnist --iid --num_channels 1 --model mlp --epochs 50 --gpu 0 --all_clients