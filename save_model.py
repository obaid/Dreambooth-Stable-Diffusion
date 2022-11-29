directory_paths = get_ipython().getoutput('ls -d logs/*')
last_checkpoint_file = directory_paths[-1] + "/checkpoints/last.ckpt"
training_images = get_ipython().getoutput('find training_images/*')
date_string = get_ipython().getoutput('date +"%Y-%m-%dT%H-%M-%S"')
max_training_steps = 5*101
class_word = "person" # typical uses are "man", "person", "woman"
token = "GS"

file_name = date_string[-1] + "_" + project_name + "_" + str(len(training_images)) + "_training_images_" +  str(max_training_steps) + "_max_training_steps_" + token + "_token_" + class_word + "_class_word.ckpt"

file_name = file_name.replace(" ", "_")

get_ipython().system('mkdir -p trained_models')
get_ipython().system('mv "{last_checkpoint_file}" "trained_models/{file_name}"')

print("Download your trained model file from trained_models/" + file_name + " and use in your favorite Stable Diffusion repo!")
