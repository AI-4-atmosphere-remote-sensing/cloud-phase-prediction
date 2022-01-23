# cloud-phase-prediction
This repository provides an end-to-end deep domain adaptation with domain mapping and correlation alignment (DAMA) and apply it to classify the heterogeneous remote satellite cloud and aerosol types..

## Prerequisite
The project currently supports `python>=3.7`

### Install dependencies 
```
>> conda create -n py37 python=3.7
>> conda activate py37
>> conda install pytorch -c pytorch -c conda-forge
>> git clone https://github.com/AI-4-atmosphere-remote-sensing/cloud-phase-prediction
>> cd cloud-phase-prediction
>> pip install .
```

## Getting start
1. Run the model training: 
`python train.py --training_data_path='/umbc/rs/nasa-access/xin/cloud-phase-prediction/sample_data/training/'  --model_saving_path='/umbc/rs/nasa-access/xin/cloud-phase-prediction/saved_model/' `.

Note you can supply the training data into directory --training_data_path you define and the trained model will be save at the specified directory --model_saving_path. The sample training and testing files have been provided in the ./data/training and ./data/testing directories, respectively.  

2. Evaluate on the testing dataset: 
`python evaluate.py --testing_data_path='/umbc/rs/nasa-access/xin/cloud-phase-prediction/sample_data/testing/'  --model_saving_path='/umbc/rs/nasa-access/xin/cloud-phase-prediction/saved_model/'`.

Note you can supply the testing data into directory --testing_data_path.

## Publications
1. Xin Huang, Sahara Ali, Chenxi Wang, Zeyu Ning, Sanjay Purushotham, Jianwu Wang, Zhibo Zhang. ["Deep Domain Adaptation based Cloud Type Detection using Active and Passive Satellite Data"](https://ieeexplore.ieee.org/abstract/document/9377756). In Proceedings of the 2020 IEEE International Conference on Big Data (BigData 2020), pages 1330-1337, IEEE, 2020.
2. Xin Huang, Sahara Ali, Sanjay Purushotham, Jianwu Wang, Chenxi Wang and Zhibo Zhang. ["Deep Multi-Sensor Domain Adaptation on Active and Passive Satellite Remote Sensing Data"](http://mason.gmu.edu/~lzhao9/venues/DeepSpatial2020/papers/DeepSpatial2020_paper_14_camera_ready.pdf). In Proceedings of the 1st KDD Workshop on Deep Learning for Spatiotemporal Data, Applications, and Systems (DeepSpatial 2020).
