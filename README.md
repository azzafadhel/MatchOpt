# Learning-Surrogates-for-Offline-Black-Box-Optimization-via-Gradient-Matching
This is the official implementation of the paper "Learning Surrogates for Offline Black-Box Optimization via Gradient Matching" published in ICML 2024

Generating the Summarized Data
To generate the summarized_data.dat file, follow these steps:

Run the Gradient Ascent Command: For all combinations of baselines and tasks, run the following command:
gradient-ascent ant --local-dir ~/pycharm/mbo-cache/gradient-ascent-ant --cpus 16 --gpus 2 --num-parallel 4 --num-samples 4

This command performs a gradient ascent operation on the ‘ant’ task. The --local-dir flag specifies the directory where the operation’s cache is stored. The --cpus and --gpus flags specify the number of CPUs and GPUs to be used, respectively. The --num-parallel flag specifies the number of parallel processes to be run, and the --num-samples flag specifies the number of samples to be generated in each process.

Run the Data Grabber Script: Once you have all the combinations of baselines and tasks stored, run the following command:
python3 data_grabber.py

This will output the summarized_data.dat file.

Model Training and Search

Train the Surrogate Model and Perform Gradient Ascent: Run the following command:
python3 gm_surrogate_traj_sampling.py

This will train the surrogate model and perform the search via gradient ascent. At the end, you will have y_sol of 128 samples from which you can grab the max, min, mean, and median values.
