
## Prerequisites
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- pytorch



- Train the model:
python train.py --dataroot ../your_data_dir --name yourname --norm batch --batch_size 16 --save_epoch_freq 100 --checkpoints_dir ../your_checkpoints_dir --num_threads 12 --netG unet_new --model pix2pixMy --freq_weight 1 --vgg_weight 1 --ssim_weight 0 --netD multi_D --num_D 3

- Test the model:
python test.py --dataroot ../your_data_dir --name yourname --checkpoints_dir ../your_checkpoints_dir --norm batch --num_test yourtestnum --results_dir ../your_result_dir --netG unet_new --model pix2pixMy

- Evaluate the modal 
python ./metric/eval.py --path_fake ../your_fake_dir --path_real ../your_real_dir --output_name ./your_result_dir



## Acknowledgments
Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
