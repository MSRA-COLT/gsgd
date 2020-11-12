python main.py --model plain34 --dataset cifar --epochs 160 --decay multistep --optimizer gsgd --ec_opt ec --inner_opt sgd --seed 1 --wd 0 --lr 1.0 --momentum 0 --batch_size 128 --log_interval 100 --check_point_name savedmodel/plain_1/plain_1-cifar-plain34-gsgd-ec-sgd-1.0-128-0-01.0-1.0-0.01-160-10-101/plain_1-cifar-plain34-gsgd-ec-sgd-1.0-128-0-160-1-101      | tee outputcnn/plain_1/plain_1-cifar-plain34-gsgd-ec-sgd-1.0-128-0-160-1-101.txt_10



python main.py --model plain34 --dataset cifar --epochs 160 --decay multistep --optimizer sgd --ec_opt bp --inner_opt sgd --seed 1 --wd 0 --lr 1.0 --momentum 0 --batch_size 128 --log_interval 100 --check_point_name savedmodel/plain_1/plain_1-cifar-plain34-sgd-bp-sgd-1.0-128-0-01.0-1.0-0.01-160-10-101/plain_1-cifar-plain34-sgd-bp-sgd-1.0-128-0-160-1-101      | tee outputcnn/plain_1/plain_1-cifar-plain34-sgd-bp-sgd-1.0-128-0-160-1-101.txt_10

wait