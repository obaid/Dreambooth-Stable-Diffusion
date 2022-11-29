# Training

dataset="person_ddim"

# This isn't used for training, just to help you remember what your trained into the model.
project_name = "gauhar"

# MAX STEPS
# How many steps do you want to train for?
max_training_steps = 5*101

# Match class_word to the category of the regularization images you chose above.
class_word = "person" # typical uses are "man", "person", "woman"

# This is the unique token you are incorporating into the stable diffusion model.
token = "GS"

reg_data_root = "/home/resemble/Development/obaid/dreambooth/Dreambooth-Stable-Diffusion/regularization_images/" + dataset

get_ipython().system('rm -rf training_images/.ipynb_checkpoints')
get_ipython().system('python "main.py"   --base configs/stable-diffusion/v1-finetune_unfrozen.yaml   -t   --actual_resume "model.ckpt"   --reg_data_root "{reg_data_root}"   -n "{project_name}"   --gpus 0,   --data_root "/home/resemble/Development/obaid/dreambooth/Dreambooth-Stable-Diffusion/training_images"   --max_training_steps {max_training_steps}   --class_word "{class_word}"   --token "{token}"   --no-test')
