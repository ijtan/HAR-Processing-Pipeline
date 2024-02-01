# Data Processing Pipeline
A human activity recognition data processing pipeline dedicated at transforming triaxial smartphone data into a 580+ feature vector, usable by classifiers for Human Activity Recognition.


# Overview
The HAR-Processing-Pipeline is dedicated to the processing of triaxial smartphone data for HAR. It provides a comprehensive solution for data augmentation and dataset orchestration, aiming to facilitate the transformation of raw sensor data into a feature-rich format suitable for HAR classification.
## Features
- Triaxial smartphone data processing
- Feature vector generation
- Dataset orchestration for HAR
- End-to-end pipeline for HAR augmentation

# Details
![image](https://github.com/ijtan/HAR-Processing-Pipeline/assets/11772153/e8e4e616-211a-42c2-803e-d69dd79a560d)

## Dataset
An android application was developed to ease the process of collecting and labelling movement data. 
7 entries at 50hz were taken namely: 
- Triaxial Linear Acceleration
- Triaxial Angular Acceleration
- Time.


## Data proportions

![image](https://github.com/ijtan/HAR-Processing-Pipeline/assets/11772153/2f1a1f45-9550-4d4e-9595-e47529aa2c93)

Sliding Window Technique

![image](https://github.com/ijtan/HAR-Processing-Pipeline/assets/11772153/114eb13e-ad6d-49dc-bf18-4272b347e10f)


## Performance
GNB = Gaussian Naive Bayes
SGD = Stochastic Gradient Descent
SVM = Support Vector Machine

OD = Our Data

## SVM:

![image](https://github.com/ijtan/HAR-Processing-Pipeline/assets/11772153/7b94e372-c7c4-4280-a509-a00fbb6fe1e0)
![image](https://github.com/ijtan/HAR-Processing-Pipeline/assets/11772153/23f8569d-cfd7-4dd1-8bb8-bab0961d44c7)

## CNN
![image](https://github.com/ijtan/HAR-Processing-Pipeline/assets/11772153/611b375f-26e5-48db-90f8-354be277b817)
![image](https://github.com/ijtan/HAR-Processing-Pipeline/assets/11772153/7e7d3417-f2b3-4adf-aadb-756102d50930)

