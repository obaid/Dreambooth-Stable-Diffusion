#!/usr/bin/env python
# coding: utf-8

# # Dreambooth
# ### Notebook implementation by Joe Penna (@MysteryGuitarM on Twitter) - Improvements by David Bielejeski
# 
# ### Instructions
# - Sign up for RunPod here: https://runpod.io/?ref=n8yfwyum
#     - Note: That's my personal referral link. Please don't use it if we are mortal enemies.
# 
# - Click *Deploy* on either `SECURE CLOUD` or `COMMUNITY CLOUD`
# 
# - Follow the rest of the instructions in this video: https://www.youtube.com/watch?v=7m__xadX0z0#t=5m33.1s
# 
# Latest information on:
# https://github.com/JoePenna/Dreambooth-Stable-Diffusion

# ## Build Environment

# In[ ]:


# If running on Vast.AI, copy the code in this cell into a new notebook. Run it, then launch the `dreambooth_runpod_joepenna.ipynb` notebook from the jupyter interface.
# get_ipython().system('git clone https://github.com/JoePenna/Dreambooth-Stable-Diffusion')


# In[ ]:


# BUILD ENV
get_ipython().system('pip install omegaconf')
get_ipython().system('pip install einops')
get_ipython().system('pip install pytorch-lightning==1.6.5')
get_ipython().system('pip install test-tube')
get_ipython().system('pip install transformers')
get_ipython().system('pip install kornia')
get_ipython().system('pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers')
get_ipython().system('pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip')
get_ipython().system('pip install setuptools==59.5.0')
get_ipython().system('pip install pillow==9.0.1')
get_ipython().system('pip install torchmetrics==0.6.0')
get_ipython().system('pip install -e .')
get_ipython().system('pip install protobuf==3.20.1')
get_ipython().system('pip install gdown')
get_ipython().system('pip install -qq diffusers["training"]==0.3.0 transformers ftfy')
get_ipython().system('pip install -qq "ipywidgets>=7,<8"')
get_ipython().system('pip install huggingface_hub')
get_ipython().system('pip install ipywidgets==7.7.1')
get_ipython().system('pip install captionizer==1.0.1')


# In[ ]:


get_ipython().system('gdown "1FXaXkbgOYT3hQHMn9Own-16YT0SEH1He&confirm=t"')
get_ipython().system('mv sd-v1-4-full-ema.ckpt model.ckpt')

# # Regularization Images (Skip this section if you are uploading your own or using the provided images)

# Training teaches your new model both your token **but** re-trains your class simultaneously.
# 
# From cursory testing, it does not seem like reg images affect the model too much. However, they do affect your class greatly, which will in turn affect your generations.
# 
# You can either generate your images here, or use the repos below to quickly download 1500 images.

# # Download pre-generated regularization images
# We've created the following image sets
# 
# `man_euler` - provided by Niko Pueringer (Corridor Digital) - euler @ 40 steps, CFG 7.5
# `man_unsplash` - pictures from various photographers
# `person_ddim`
# `woman_ddim` - provided by David Bielejeski - ddim @ 50 steps, CFG 10.0
# `person_ddim` is recommended

# In[ ]:


#Download Regularization Images

dataset="person_ddim" #@param ["man_euler", "man_unsplash", "person_ddim", "woman_ddim", "blonde_woman"]
get_ipython().system('git clone https://github.com/djbielejeski/Stable-Diffusion-Regularization-Images-{dataset}.git')

get_ipython().system('mkdir -p regularization_images/{dataset}')
get_ipython().system('mv -v Stable-Diffusion-Regularization-Images-{dataset}/{dataset}/*.* regularization_images/{dataset}')


# # Upload your training images
# Upload 10-20 images of someone to
# 
# ```
# /home/resemble/Development/obaid/dreambooth/Dreambooth-Stable-Diffusion/training_images
# ```
# 
# WARNING: Be sure to upload an *even* amount of images, otherwise the training inexplicably stops at 1500 steps.
# 
# *   2-3 full body
# *   3-5 upper body
# *   5-12 close-up on face
# 
# The images should be:
# 
# - as close as possible to the kind of images you're trying to make

# ## Training
# 
# If training a person or subject, keep an eye on your project's `logs/{folder}/images/train/samples_scaled_gs-00xxxx` generations.
# 
# If training a style, keep an eye on your project's `logs/{folder}/images/train/samples_gs-00xxxx` generations.

# In[ ]:


# # Training

# # This isn't used for training, just to help you remember what your trained into the model.
# project_name = "gauhar"

# # MAX STEPS
# # How many steps do you want to train for?
# max_training_steps = 5*101

# # Match class_word to the category of the regularization images you chose above.
# class_word = "person" # typical uses are "man", "person", "woman"

# # This is the unique token you are incorporating into the stable diffusion model.
# token = "Aziz Ansari"

# reg_data_root = "/home/resemble/Development/obaid/dreambooth/Dreambooth-Stable-Diffusion/regularization_images/" + dataset

# get_ipython().system('rm -rf training_images/.ipynb_checkpoints')
# get_ipython().system('python "main.py"   --base configs/stable-diffusion/v1-finetune_unfrozen.yaml   -t   --actual_resume "model.ckpt"   --reg_data_root "{reg_data_root}"   -n "{project_name}"   --gpus 0,   --data_root "/home/resemble/Development/obaid/dreambooth/Dreambooth-Stable-Diffusion/training_images"   --max_training_steps {max_training_steps}   --class_word "{class_word}"   --token "{token}"   --no-test')


# ## Copy and name the checkpoint file

# In[ ]:


# Copy the checkpoint into our `trained_models` folder

# directory_paths = get_ipython().getoutput('ls -d logs/*')
# last_checkpoint_file = directory_paths[-1] + "/checkpoints/last.ckpt"
# training_images = get_ipython().getoutput('find training_images/*')
# date_string = get_ipython().getoutput('date +"%Y-%m-%dT%H-%M-%S"')
# file_name = date_string[-1] + "_" + project_name + "_" + str(len(training_images)) + "_training_images_" +  str(max_training_steps) + "_max_training_steps_" + token + "_token_" + class_word + "_class_word.ckpt"

# file_name = file_name.replace(" ", "_")

# get_ipython().system('mkdir -p trained_models')
# get_ipython().system('mv "{last_checkpoint_file}" "trained_models/{file_name}"')

# print("Download your trained model file from trained_models/" + file_name + " and use in your favorite Stable Diffusion repo!")


# # Big Important Note!
# 
# The way to use your token is `<token> <class>` ie `joepenna person` and not just `joepenna`

# ## Generate Images With Your Trained Model!

# In[ ]:


# get_ipython().system('python scripts/stable_txt2img.py   --ddim_eta 0.0   --n_samples 1   --n_iter 4   --scale 7.0   --ddim_steps 50   --seed 4567999   --ckpt "/home/resemble/Development/obaid/dreambooth/Dreambooth-Stable-Diffusion/trained_models/{file_name}"   --prompt "portrait of Aziz Ansari person ultrarealistic, 4k, shot on nikon DSLR"')


# In[ ]:




