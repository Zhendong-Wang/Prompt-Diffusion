name=$1  # experiment name
nnode=$2  # number of nodes for multi-node training, default=1

python tool_add_control.py 'path to your stable diffusion checkpoint, e.g., /.../v1-5-pruned-emaonly.ckpt' ./models/control_sd15_ini.ckpt

python train.py --name ${name} --gpus=8 --num_nodes=${nnode} \
       --logdir 'your logdir path' \
       --data_config './models/dataset.yaml' --base './models/cldm_v15.yaml' \
       --sd_locked

