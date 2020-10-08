import os


max_sequence_len = 512

num_classes = 3

class enum_definations:
    neutral_label  = 0
    positive_label = 1
    negative_label = 2


trainer_args = dict(

    accumulate_grad_batches= 4,
    fast_dev_run= True,
    max_epochs = 5,
    default_root_dir= os.path.join(os.getcwd(),'checkpoints')
)