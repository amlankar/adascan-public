python adascan.py -gpus /gpu:0,/gpu:1,/gpu:2 \
	          -data_dir /path/to/npz_files/ \
		  -split_dir /path/to/split_files/ \
		  -data_list name_of_split_file_to_use \
		  -vgg_path /path/to/vgg_npy \
		  -batch_size batch_size \
		  -input rgb/flow -num_channels 3/20 \
		  -name checkpoints/path/to/model_file \
		  -flip True \
		  -mode test \
		  -num_data_threads 16 \

#NOTE: To save a simpler version of the model (without the gradients and losses)
#you can run this code with -mode save, remember to set batch_size to 1 for it to
#work with demo.py. You can change demo.py to work with any batch size of your choice

#For full list of options, see adascan.py

#NOTE: for -flip False and -save False, please delete the lines instead of
#writing False. Python interprets it as a string and sets it to the boolean True
