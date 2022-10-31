# Cloud Phase Prediction with Multi-Sensor Domain Adaptation 
This repository provides an end-to-end deep domain adaptation with domain mapping and correlation alignment (DAMA) and apply it to classify the heterogeneous remote satellite cloud and aerosol types.

## Prerequisite
The project currently supports `python>=3.7`

### Install dependencies
```
>> conda create -n cloud-phase-prediction-env -c conda-forge python=3.7 pytorch h5py pyhdf
>> conda activate cloud-phase-prediction-env
>> git clone https://github.com/AI-4-atmosphere-remote-sensing/cloud-phase-prediction
>> cd cloud-phase-prediction
>> pip install .
```

## Data preprocessing
Note: the example npz files were provided in [example](https://github.com/AI-4-atmosphere-remote-sensing/cloud-phase-prediction/tree/main/example) folder, you can skip Data preprocessing steps and use the provided preprocessed [example](https://github.com/AI-4-atmosphere-remote-sensing/cloud-phase-prediction/tree/main/example) files in training and evaluating the prediction below. 

1. Use [satellite_collocation](https://github.com/AI-4-atmosphere-remote-sensing/satellite_collocation) to download and collocate the CALIPSO and VIIRS satellite data to generate the collocated h5 files. 
2. Read Calipso and VIIRS features from individual h5 files and generate combined h5 files with collocated timestamps and cloud phases labels. For example, to generate collocated data files of 2013:
```
>> python data_preprocess/collocate_aerosol_free_data.py 2013
```
3. Read Calipso and VIIRS features and labels from multile h5 files and generate a single npz file for a given year. For example, to generate a single npz data file of 2013:
```
>> python data_preprocess/read_calviirs_data_in_npz.py 2013 
```

## Train and evaluate the DAMA-WL prediction model
1. Run the model training: 
```
>> python train.py --training_data_path='./example/training_data/'  --model_saving_path='./saved_model/'
```

Note you can supply the training data into directory --training_data_path you define and the trained model and scaler will be save at the specified directory --model_saving_path. The sample training and testing files have been provided in the ./data/training and ./data/testing directories, respectively.  

2. Evaluate on the testing dataset: 
```
>> python evaluate.py --testing_data_path='./example/testing_data/'  --model_saving_path='./saved_model/'
```

## Train and evaluate the VDAM prediction model
1. Run the model training: 
```
>> python train_vdam.py --training_data_path='./example/training_data/'  --model_saving_path='./saved_model/vdam/'
```

Note you can supply the training data into directory --training_data_path you define and the trained model and scaler will be save at the specified directory --model_saving_path. The sample training and testing files have been provided in the ./data/training and ./data/testing directories, respectively.  

2. Evaluate on the testing dataset: 
```
>> python evaluate_vdam.py --testing_data_path='./example/testing_data/'  --model_saving_path='./saved_model/vdam/'
```

Note you can supply the testing data in directory --testing_data_path and the trained model and scaler in directory --model_saving_path.

## Methodology
Domain adaptation techniques have been developed to handle data from multiple sources or domains. Most existing domain adaptation models assume that source and target domains are homogeneous, i.e., they have the same feature space. Nevertheless, many real world applications often deal with data from heterogeneous domains that come from completely different feature spaces. In our remote sensing application, data in source domain (from an active spaceborne Lidar sensor CALIOP onboard CALIPSO satellite) contain 25 attributes, while data in target domain (from a passive spectroradiometer sensor VIIRS onboard Suomi-NPP satellite) contain 20 different attributes. CALIOP has better representation capability and sensitivity to aerosol types and cloud phase, while VIIRS has wide swaths and
better spatial coverage but has inherent weakness in differentiating atmospheric objects on different vertical levels. To address this mismatch of features across the domains/sensors, we propose a novel end-to-end deep domain adaptation with domain mapping and correlation alignment (DAMA) to align the heterogeneous
source and target domains in active and passive satellite remote sensing data. It can learn domain invariant representation from source and target domains by transferring knowledge across these domains, and achieve additional performance improvement by incorporating weak label information into the model (DAMA-WL). The architect of DAMA is shown below: 
![architect](architect.png)

## Results
Our experiments on a collocated CALIOP and VIIRS dataset show that DAMA and DAMA-WL can achieve higher classification accuracy in predicting cloud types. In the experimental setting, both training and testing of target domain are performed using the weak labels from VIIRS. Comparing VIIRS’ weak labels with corresponding CALIOP category 2 and category 3 labels (further explained in Section V-A of our Big Data conference paper below), we get around 87% label match rate. The CALIOP label is considered as ground truth for the experiment, so we can consider the weak label is noisy of only 87% accuracy. Result table introduces two more experiments Random ForestWL and DAMA-WL that utilize weak label from VIIRS dataset. Random Forest-WL trains the model with weak label from VIIRS dataset of the second label setting, while DAMAWL adds a target classifier trained with weak label of VIIRS dataset to DAMA. Other models shown in Table II use the CALIOP label from collocated CALIOP and VIIRS pixels.
From Table II we can see DAMA-WL achieves highest accuracy 96.0% compared to the random forest models and other baseline models. We also see DAMA-WL brings additional 1.9% accuracy improvement compared to the DAMA method, which shows that the weak label does help train a better domain adaptation model in weak supervision on target domain.
![result](result.png)

## Publications
1. Xin Huang, Chenxi Wang, Sanjay Purushotham, Jianwu Wang. VDAM: VAE based Domain Adaptation for Cloud Property Retrieval from Multi-satellite Data. ACM SIGSPATIAL 2022 (Long Paper).
2. Xin Huang, Sahara Ali, Chenxi Wang, Zeyu Ning, Sanjay Purushotham, Jianwu Wang, Zhibo Zhang. ["Deep Domain Adaptation based Cloud Type Detection using Active and Passive Satellite Data"](https://ieeexplore.ieee.org/abstract/document/9377756). In Proceedings of the 2020 IEEE International Conference on Big Data (BigData 2020), pages 1330-1337, IEEE, 2020.
3. Xin Huang, Sahara Ali, Sanjay Purushotham, Jianwu Wang, Chenxi Wang and Zhibo Zhang. ["Deep Multi-Sensor Domain Adaptation on Active and Passive Satellite Remote Sensing Data"](http://mason.gmu.edu/~lzhao9/venues/DeepSpatial2020/papers/DeepSpatial2020_paper_14_camera_ready.pdf). In Proceedings of the 1st KDD Workshop on Deep Learning for Spatiotemporal Data, Applications, and Systems (DeepSpatial 2020).

