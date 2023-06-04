# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
The Bank Marketing dataset from UCI ML Repository, contains demographic data of bank clients and their responses (Yes or No) to direct phone marketing campaigns of direct term deposit products. The classification goal is to predict if the client will subscribe a term deposit. Therefore the input variables are columns representing client demographics and the output variable is the y column representing has the client subscribed to a term deposit (binary Yes or No).

The best performing model is an ensemble model VotingEnsemble produced by the AutomML run. It has an accuracy rate of 91.803% vs 90.61% by the HyperDrive assisted Scikit-learn LogicRegression model.

## Scikit-learn Pipeline
The pipeline architecture consists of a python training script (train.py), a tabular dataset downloaded from UCI ML Repository, a Scikit-learn Logistic Regression Algorithm connected to the Azure HyperDrive, a hyperparameter tuning engine, to produce a HyperDrive classifier. The training run was orchestrated by a Jupyter Notebook hosted on a compute instance. The diagram below (Image credit: Udacity MLEMA Nanodegree) presents a logical view of the architecture.

![image](https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/d872119b-e159-403c-b4a9-25f1aff754fb)

## Dataset
The dataset was programmatically (using the train.py script) downloaded from the web, split into train and test sets using Sckit-learn train_test_split utility

### Benefits of the parameter sampler chosen
The random parameter sampler for HyperDrive supports discrete and continuous hyperparameters, as well as early termination of low-performance runs. It is simple to use, eliminates bias and increases the accuracy of the model.

### Benefits of the early stopping policy chosen
The early termination policy BanditPolicy for HyperDrive automatically terminates poorly performing runs and improves computational efficiency. It is based on slack factor/slack amount and evaluation interval and cancels runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.
**What are the benefits of the early stopping policy you chose?**

## AutoML
The AutoML run was executed with this AutoMLConfig settings:

automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    compute_target=compute_target,
    task='classification',
    primary_metric='accuracy',
    training_data=ds,
    label_column_name='y',
    n_cross_validations=5)
    
The best model generated from the run was a VotingEnsemble model from 24 estimators.

Best run metrics : 
{'weighted_accuracy': 0.9521390343726057, 'log_loss': 0.28968172426506567, 'precision_score_macro': 0.7969453106014988, 'precision_score_micro': 0.9180273141122914, 'recall_score_macro': 0.7806722400586181, 'balanced_accuracy': 0.7806722400586181, 'recall_score_micro': 0.9180273141122914, 'average_precision_score_micro': 0.9814981927757108, 'norm_macro_recall': 0.5613444801172364, 'AUC_macro': 0.948344099217516, 'f1_score_micro': 0.9180273141122914, 'recall_score_weighted': 0.9180273141122914, 'matthews_correlation': 0.577281680858296, 'precision_score_weighted': 0.9160166000693654, 'AUC_weighted': 0.948344099217516, 'average_precision_score_macro': 0.8294230971606277, 'AUC_micro': 0.9808449966726613, 'accuracy': 0.9180273141122914, 'f1_score_macro': 0.7883333448527374, 'f1_score_weighted': 0.9168993980826686, 'average_precision_score_weighted': 0.9564402704864866, 'confusion_matrix': 'aml://artifactId/ExperimentRun/dcid.AutoML_44dd3d6e-aa91-4c06-ba85-2d3d39e104f7_28/confusion_matrix', 'accuracy_table': 'aml://artifactId/ExperimentRun/dcid.AutoML_44dd3d6e-aa91-4c06-ba85-2d3d39e104f7_28/accuracy_table'}

## Pipeline comparison
The HyperDrive assisted Scikit-learn LogicRegression model produced a top accuracy of 90.61% as shown below:

<img width="588" alt="Screenshot 2023-06-04 at 9 35 22 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/a25d84b4-7abf-4867-94f9-fa477ffccba6">
![image](https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/093a393a-7171-417e-a375-461f75f7959d)


The AutoML generated VotingEnsemble model yielded a top accuracy of 91.80% as shown below
<img width="1189" alt="Screenshot 2023-06-04 at 9 11 23 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/0ea0baa0-c3d3-4385-a133-8c8661a0c605">

HyperDrive performed way faster than AutoML
![image](https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/fe134b9e-0820-4659-861e-67a19ca05660)


The difference in accuracy is small. However, architectuarally, there is a difference between the two. To use HyperDrive, a custom-coded machine learning model is required. Whereas to utilize AutoML, one only needs to select some paramters for AutoML config and AutoML does the rest. Additionally, AutoML offers model interpretation which is useful in understanding why a model made a certain prediction as well as getting an idea of the importance of individual features for tasks.

Here are some AutoML generated visual explanations and metrics:
<img width="554" alt="Screenshot 2023-06-04 at 9 14 08 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/cd369d60-50ef-41c0-bada-59e88ece1f94">
<img width="778" alt="Screenshot 2023-06-04 at 9 14 33 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/6e90178e-e0f7-483f-9c82-783cd311d50d">
<img width="590" alt="Screenshot 2023-06-04 at 9 15 04 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/2c3cb76d-714e-4e05-9f33-c0df0fe3ba4d">
![image](https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/081864d0-8b00-470a-ada9-94c301570a11)
<img width="505" alt="Screenshot 2023-06-04 at 9 27 25 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/50686522-f3a5-4284-849d-d2e69b3905ac">
<img width="500" alt="Screenshot 2023-06-04 at 9 28 09 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/9267e9bc-e8ed-4cd8-94c4-4c6d915ab74c">
<img width="501" alt="Screenshot 2023-06-04 at 9 27 34 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/0d888474-aad0-4c4c-b6f1-f0e119688683">
<img width="502" alt="Screenshot 2023-06-04 at 9 27 45 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/2f6f49c3-b069-4c4c-8612-bc9cbc962443">


It also talked about the data imbalance present in Data Guardrails section
<img width="1212" alt="Screenshot 2023-06-04 at 8 45 39 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/919d210f-19e9-4599-a2e3-c5ce754ce75b">


Also the pipeline used by AutoML
![image](https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/8642f958-0449-45d9-8cab-51fe4a598f39)



## Future work
1) We can experiment the Hyperdrive run to include grid sampling or bayesian sampling. This would take time but could give better results.
2) In AutoML, based on the Guardrails details, we can try to minimize the bias of data. The model tend to perform better on a balanced dataset.

## Proof of cluster clean up
![image](https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/8671b860-a315-4aed-b930-7878808f64fc)

