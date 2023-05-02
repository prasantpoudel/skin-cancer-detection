"""
                            This is a streamlit app.
                            To run this use: streamlit run main.py

"""
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# st.markdown(
#      """
#      <style>
#      .stApp {
#         background-image:url( "https://cdn.discordapp.com/attachments/432973736946171917/1102284008160763986/img3.png");
#         background-size: cover;
#      }
#      </style>
#      """,
#      unsafe_allow_html=True
# )
labels = {
    0: "actinic keratoses and intraepithelial carcinomae (Cancer)",
    1: "basal cell carcinoma (Cancer)",
    2: "benign keratosis-like lesions (Non-Cancerous)",
    3: "dermatofibroma (Non-Cancerous)",
    4: "melanoma (Cancer)",
    5: "melanocytic nevi (Non-Cancerous)",
    6: "Vascular lesion (Non-Cancerous)"
}
model = tf.keras.models.load_model('front_model_resnet.h5')
classify_model=tf.lite.Interpreter(model_path="model/InceptionResNetV2Skripsi.tflite")
classify_model.allocate_tensors()

input_details = classify_model.get_input_details()
output_details = classify_model.get_output_details()

def label_print(image):
    probs = classify_image1(image)

    # # Display the top 3 predictions
    top_3_indices = np.argsort(probs)[::-1][:3]
    st.write("Top 3 predictions:")
    for i in range(3):
        st.write("%d. %s (%.2f%%)" % (i + 1, labels[top_3_indices[i]], probs[top_3_indices[i]] * 100))
    ind=probs.argmax()
    st.write("The Most possible label Will be:",labels[ind])

def detect_skin(image):
    # Convert the image to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Apply skin color detection algorithm
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Apply morphological transformations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Count the number of skin pixels
    num_skin_pixels = cv2.countNonZero(mask)
    
    # Calculate the ratio of skin pixels to total pixels
    ratio = num_skin_pixels / (image.shape[0] * image.shape[1])
    
    return ratio

def resize_image(image):
    # Resize the image to 150x150 pixels
    resized_image = tf.image.resize(image, [150, 150])
    return resized_image.numpy()

def classify_image1(image):
    # Pre-process the input image
    resized_image = resize_image(image)
    input_data = np.expand_dims(resized_image, axis=0).astype(np.float32)
    classify_model.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    with st.spinner('Classifying...'):
        classify_model.invoke()

    # Get the output probabilities
    output_data = classify_model.get_tensor(output_details[0]['index'])
    return output_data[0]
def classify_image(img, model):
    image=img
    img = img.resize((224, 224))  # Resize the image to match the model input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        st.write("The image is classified as class Cancer")
        label_print(image)

    else:
        st.write("The image is classified as class non cancer")
        if st.button('Bypass the non cancer'):
            label_print(image)



# Load the pre-trained model
model = tf.keras.models.load_model('front_model_resnet.h5')
classify_model=tf.lite.Interpreter(model_path="InceptionResNetV2Skripsi.tflite")
classify_model.allocate_tensors()


# Define the Streamlit app
st.title("Skin Cancer Detection")
st.sidebar.title('Input Image')
st.sidebar.markdown('Upload an image of a skin lesion to make a prediction.')
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","HEIC"])
if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    image = cv2.resize(image, (500, 500))
    # image = cv2.resize(image, (224, 224))
        
    ratio = detect_skin(image)
        
    # Display the result
    # st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Ratio of skin pixels to total pixels: {ratio:.2f}")
    if ratio > 0.4:
        st.write("The image contains skin.")
        image = Image.open(uploaded_file)
        st.image(image, width=300)
        st.write("")
        st.write("Classifying...")
        label = classify_image(image, model)
    else:
        st.write("please upload the skin image")
        # some time the skin image get the ratio less than 0.5 then manual overite
        if st.button('Bypass the skin validation'):
            image = Image.open(uploaded_file)
            st.image(image, width=300)
            st.write("")
            st.write("Classifying...")
            label = classify_image(image, model)

    