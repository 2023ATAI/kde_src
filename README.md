### Sampling-Weighted Sensitive Learning Strategy
To address the challenge of data imbalance in soil moisture (SM) prediction, we propose a Sampling-Weighted Sensitive Learning Strategy. This method enhances model sensitivity to rare but important samples by assigning adaptive weights during training. By emphasizing underrepresented data regions, the strategy improves model generalization and short-term prediction accuracy. It has been effectively integrated into multiple deep learning models, including LSTM, BiLSTM, and GRU, and demonstrated superior performance compared to baseline approaches.

### Dataset
We use the LandBench1.0 dataset proposed by Li et al. and the data-driven land surface variables (LSVs) prediction toolbox.

The LSVs benchmark dataset is hosted here(https://doi.org/10.11888/Atmos.tpdc.300294)
The prediction toolbox is hosted here(https://github.com/2023ATAI/LandBench1.0)

### Requirements
LandBench works in Python3.9.13
In order to use the LandBench successfully, the following site-packages are required:

pytorch 1.13.1
pandas 1.4.4
numpy 1.22.0
scikit-learn 1.0.2
scipy 1.7.3
matplotlib 3.5.2
xarray 2023.1.0
netCDF4 1.6.2
The latest LandBench1.0 can work in

linux-Ubuntu 18.04.6

### Prepare Config File
Usually, we use the config file in model training, testing and detailed analyzing.

The config file contains all necessary information, such as path,data,model, etc.

The config file of our work is LandBench1.0/src/config.py

### Process data and train model
Run the following command in the directory of LandBench1.0/src/ to process data and start training.
python main.py 

### Detailed analyzing
Run the following command in the directory of LandBench1.0/src/ to get detailed analyzing.
python postprocess.py 
python post_test.py 
