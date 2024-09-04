# Music_Therapy_for_people_with_ASD_using_Emotion_Detection
### Deep learning models used 
CNN, ResNet50V2, VGG-19 

### Technologies Used 
Keras, pandas, Streamlit, Matplotlib

### Overview
This project develops a specialized facial emotion recognition system tailored for individuals with autism spectrum disorder (ASD) using a modified ResNet50V2 convolutional neural network architecture pre-trained on the ImageNet dataset. This choice was made because ResNet50V2 allows the capturing of complex features from facial images. It accurately classifies emotional expressions, especially for ASD individuals who exhibit nuanced emotional responses. We classified the train dataset of facial images into seven states – happy, sad, angry, neutral, fear, disgust and surprise, for enabling precise emotion detection for ASD individuals. We have enhanced the model’s robustness and mitigated overfitting by employing image data augmentation and advanced regularization techniques. The model is trained and validated with real-time data augmentation and a dynamic learning rate adjustments for optimal performance. Post recognition, the model interacts with a music recommendation module that utilizes a curated dataset of songs labelled with respective emotional states. Based on the recognized emotion, the system recommends a song from the dataset which is then redirected to YouTube. This integration of emotion sensitive media recommendation aims to support emotional therapeutic interventions for those on the autism spectrum.

### Files
'_emotion-resnet50v2raf-baocao.ipynb_' - consists of the main model + backend

'_spotify_playlist_tracks.xlsx_' - dataset consisting of data scraped from spotify with keywords such as "autism", "happy", "sad", "angry", "neutral", "fear", "disgust" and "surprise"

'_main.py_' - code for streamlit frontend
