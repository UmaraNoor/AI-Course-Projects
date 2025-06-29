Streamlit app link: https://plant-disease-detector-jhzgjuhz8j7brdvqujmmfq.streamlit.app/

GitHub Link: https://github.com/Roobi-hayat/plant-disease-detector/tree/main

dataset Link: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset?select=color

AI-Powered Plant Disease Detection System

This system diagnoses plant diseases from leaf images using deep learning technology.
Technology Used:
•	TensorFlow for building the deep learning model
•	Streamlit for creating a user-friendly web interface

Plant Support:
•	Provides instant diagnosis with treatment and prevention suggestions
•	Supports 9 common plant types:
  Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, and Tomato 
  
Dataset:
•	Source: PlantVillage dataset (on Kaggle)
•	Contains 40,000 images
•	Covers 9 plant types and 24 disease classes

Algorithm & Implementation:
Transfer Learning:
•	Used MobileNetV2 (pretrained on ImageNet) for feature extraction.
•	MobileNetV2 breaks down images into 1,280 distinct features.
Performance:
•	Achieved 96.4% validation accuracy

Backend:
Uses SQLite database to store treatment and prevention info
