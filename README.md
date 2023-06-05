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
Data is fetched from here: https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv

The best performing model is an ensemble model VotingEnsemble produced by the AutomML run. It has an accuracy rate of 91.803% vs 90.61% by the HyperDrive assisted Scikit-learn LogicRegression model.

## Scikit-learn Pipeline
The pipeline architecture consists of a python training script (train.py), a tabular dataset downloaded from UCI ML Repository, a Scikit-learn Logistic Regression Algorithm connected to the Azure HyperDrive, a hyperparameter tuning engine, to produce a HyperDrive classifier. The training run was orchestrated by a Jupyter Notebook hosted on a compute instance. The diagram below (Image credit: Udacity MLEMA Nanodegree) presents a logical view of the architecture.

![image](https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/dc5e5c13-f563-44cf-85e3-108ee2af82f6)

### Dataset
The dataset was programmatically (using the train.py script) downloaded from the web, split into train and test sets using Sckit-learn train_test_split utility.

The Bank Marketing dataset from UCI ML Repository, contains demographic data of bank clients and their responses (Yes or No) to direct phone marketing campaigns of direct term deposit products. The classification goal is to predict if the client will subscribe a term deposit. Therefore the input variables are columns representing client demographics (age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, campaign, pdays, previous, poutcome, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed) and the output variable is the y column representing has the client subscribed to a term deposit (binary Yes or No).

Data cleaning steps:

Removing NAs from the dataset.
One-hot encoding job titles, contact, and education variables.
Encoding a number of other categorical variables.
Encoding months of the year.
Encoding the target variable.

We can also generate data profile to look at feature metrics indivdually like this:
<img width="1130" alt="image" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/46be0dde-31a9-4f83-90e3-9389ccfd36ea">


The classification method used here is logistic regression. Logistic regression uses a fitted logistic function and a threshold. The parameters available within the training script are C (which indicates the regularization strength i.e. preference for sparser models) and maximum number of iterations.

### Benefits of the parameter sampler chosen
The random parameter sampler for HyperDrive supports discrete and continuous hyperparameters, as well as early termination of low-performance runs. It is simple to use, eliminates bias and increases the accuracy of the model.

### Benefits of the early stopping policy chosen
The early termination policy BanditPolicy for HyperDrive automatically terminates poorly performing runs and improves computational efficiency. It is based on slack factor/slack amount and evaluation interval and cancels runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.

Best run metrics : {'Regularization Strength:': 10.0, 'Max iterations:': 150, 'Accuracy': 0.9061665452779801}

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
![image](https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/01c324a4-06f2-44fa-995c-2b869a4117bc)
![image](https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/ff82d2ec-f00d-441e-860e-2a512b9b5b41)


The AutoML generated VotingEnsemble model yielded a top accuracy of 91.80% as shown below
<img width="1189" alt="Screenshot 2023-06-04 at 9 11 23 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/da7a115b-7113-4495-8228-6e218e84fe33">


HyperDrive performed way faster than AutoML
<img width="1169" alt="Screenshot 2023-06-04 at 9 36 37 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/355cd5c0-443c-4d1b-b3d1-136142330770">


The difference in accuracy is small. However, architectuarally, there is a difference between the two. To use HyperDrive, a custom-coded machine learning model is required. Whereas to utilize AutoML, one only needs to select some paramters for AutoML config and AutoML does the rest. Additionally, AutoML offers model interpretation which is useful in understanding why a model made a certain prediction as well as getting an idea of the importance of individual features for tasks.


Here are some AutoML generated visual explanations and metrics:
<img width="554" alt="Screenshot 2023-06-04 at 9 14 08 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/b6fcab6f-d15f-4956-8ff1-a01057161201">
<img width="778" alt="Screenshot 2023-06-04 at 9 14 33 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/89c9f61f-f1e1-4f22-9dbd-555030ef3e95">
<img width="590" alt="Screenshot 2023-06-04 at 9 15 04 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/bf86b234-d859-45a8-b03c-92f9172804c3">
<img width="762" alt="Screenshot 2023-06-04 at 9 18 02 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/570429cc-4cf5-477e-a4fe-535c08ed8e16">
<img width="505" alt="Screenshot 2023-06-04 at 9 27 25 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/4d39cd94-e0c2-4368-88fa-b8b195de9d51">
<img width="501" alt="Screenshot 2023-06-04 at 9 27 34 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/b0a3b9aa-3fde-41cf-a004-4e8a70650b22">
<img width="502" alt="Screenshot 2023-06-04 at 9 27 45 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/56362d72-f21b-4192-94ee-b02792f4ec9b">
<img width="500" alt="Screenshot 2023-06-04 at 9 28 09 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/e46723cc-5ac1-4d60-a76f-0a2eb0a527c4">


It also talked about the data imbalance present in Data Guardrails section:
<img width="1212" alt="Screenshot 2023-06-04 at 8 45 39 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/f720cf73-e745-4232-a1c0-a1186d4f9e68">


Also the pipeline used by AutoML:
<img width="1151" alt="Screenshot 2023-06-04 at 9 18 38 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/fea52d1f-5bf2-4c2f-a231-f8acf114789a">


## Future work
1) We can experiment the Hyperdrive run to include grid sampling or bayesian sampling. This would take time but could give better results.
2) In AutoML, based on the Guardrails details, we can try to minimize the bias of data. The model tend to perform better on a balanced dataset.

## Proof of cluster clean up
<img width="415" alt="Screenshot 2023-06-04 at 9 31 51 PM" src="https://github.com/Nikhil-Bansal19/MLE-NanoDegree/assets/47290347/04e79f41-69e7-499c-9074-fb6b1bdc06f2">

