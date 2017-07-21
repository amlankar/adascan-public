python adascan.py -gpus /gpu:0,/gpu:1,/gpu:2 \
	          -data_dir /path/to/npz_files/ \
		  -split_dir /path/to/split_files/ \
		  -data_list name_of_split_file_to_use \
		  -vgg_path /path/to/vgg_npy \
		  -batch_size batch_size \
		  -input rgb/flow -num_channels 3/20 \
		  -keep_prob adascan_output_keep_prob \
		  -vgg_keep_prob vgg_output_keep_prob \
		  -logdir ./log/ \
		  -name model_name \
		  -flip True \
		  -mode train \
		  -save True \
		  -num_data_threads 16 \
		  -save_freq 500 \
		  -print_freq 10

#For full list of options, see adascan.py

#NOTE: for -flip False and -save False, please delete the lines instead of
#writing False. Python interprets it as a string and sets it to the boolean True
