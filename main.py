import streamlit as st
from PIL import Image
import subprocess, threading, pickle, vlc
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing import image as keras_image
import random
import urllib.parse
from IPython.display import YouTubeVideo
from keras.preprocessing import image

def loadModel():
    return load_model('ResNet50V2_Model.h5')
    # return load_model('ResNet50v2.h5')

def load_data():
    return pd.read_excel('E:/6th_sem_stuff/tdl/project/spotify_playlist_tracks.xlsx', engine='openpyxl')

def predictEmotion(image_path):
    # st.write("In the predict emotion function!!!!")
    #image pre-processing
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) #adding batch expansion
    img_array /= 255 #normalize the image

    #predicting emotion
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    model = loadModel()
    predictions = model.predict(img_array)
    predicted_label_index = np.argmax(predictions)
    predicted_label = emotion_labels[predicted_label_index]
    st.write("Predicted Emotion is: ", predicted_label)

    return predicted_label

def play_song(predicted_emotion):
    # Load the data from the Excel file
    songs_data = load_data()

    # Filter based on emotion
    filtered_data = songs_data[songs_data['emotion'] == predicted_emotion]

    # Now randomly pick a song
    if not filtered_data.empty:
        random_song_index = random.randint(0, len(filtered_data) - 1)
        random_song = filtered_data.iloc[random_song_index]

        # Print the randomly picked song
        # st.write("Custom picked song based on predicted emotion:", predicted_emotion)
        st.write("Track Name:", random_song['track name'])
        # st.write("Artist:", random_song['artist'])

        # Search for the song on YouTube
        song_query = f"{random_song['track name']} {random_song['artist']} audio"
        encoded_query = urllib.parse.quote(song_query)
        search_url = f"https://www.youtube.com/results?search_query={encoded_query}"
        # Display the YouTube video
        st.markdown(f'<a href="{search_url}" target="_blank">Click here to play "{random_song["track name"]}" on YouTube</a>', unsafe_allow_html=True)
            # st.markdown(f'<a href="{search_url}" </a>', unsafe_allow_html=True)

    else:
        st.write("No songs found for predicted emotion:", predicted_emotion)

def main():
    st.title("Emotion Based Music Player")
    #now to upload the image
    uploaded_file = st.file_uploader("Upload an image to detect emotion", type=["jpeg","jpg","png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image,caption="Uploaded Image")
        emotion = predictEmotion(uploaded_file)

        play_song(emotion)

if __name__ == "__main__":
    main()