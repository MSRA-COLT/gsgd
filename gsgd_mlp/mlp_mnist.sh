mkdir -p outputmlp/mnist



CUDA_VISIBLE_DEVICES=0 python main_mnist.py --data mnist --model mnistmlp --lr 0.1 --opt ec  --epoch 300 --decay exp --power 0.01  --batch_size 64 --seed 0 --cuda | tee outputmlp/mnist/simple_den_mnist_ec_0.1_64_expdecay_300.txt_0 &



CUDA_VISIBLE_DEVICES=5 python main_mnist.py --data mnist --model mnistmlp --lr 0.1 --opt bp  --epoch 300 --decay exp --power 0.01  --batch_size 64 --seed 5 --cuda | tee outputmlp/mnist/simple_den_mnist_bp_0.1_64_expdecay_300.txt_5 &
