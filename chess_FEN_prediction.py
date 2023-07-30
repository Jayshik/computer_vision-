import streamlit as st

import matplotlib.image as mpimg
from keras.models import load_model
from skimage import io, transform, util
import numpy as np

def process_image(img):
    downsample_size = 200
    img_read = io.imread(img)
    img_read = transform.resize(img_read, (downsample_size, downsample_size), mode='constant')

    # Make the image square by adding a border
    border_size = abs(img_read.shape[0] - img_read.shape[1]) // 2
    if img_read.shape[0] < img_read.shape[1]:
        img_read = np.pad(img_read, ((border_size, border_size), (0, 0), (0, 0)), mode='constant')
    else:
        img_read = np.pad(img_read, ((0, 0), (border_size, border_size), (0, 0)), mode='constant')

    # Split the image into tiles
    square_size = img_read.shape[0] // 8
    tiles = np.empty((64, 25, 25, 3))
    for i in range(8):
        for j in range(8):
            tile = img_read[i * square_size: (i + 1) * square_size, j * square_size: (j + 1) * square_size, :]
            tiles[i * 8 + j] = transform.resize(tile, (25, 25, 3), mode='constant')

    return tiles


def fen_from_onehot(one_hot):
    piece_symbols = ' prbnkqPRBNKQ'
    output = ''
    for j in range(8):
        for i in range(8):
            piece_idx = np.argmax(one_hot[j][i])
            output += piece_symbols[piece_idx]
        if j < 7:
            output += '-'
    return output

model = load_model("trained_model.h5")

st.title("Chess Piece Position Prediction")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = mpimg.imread(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Process the image and make a prediction
    processed_image = process_image(uploaded_file)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    prediction = model.predict(processed_image)
    predicted_fen = fen_from_onehot(prediction)

    # Display the predicted FEN notation
    st.write(f"Predicted FEN Notation: {predicted_fen}")
