import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Load the scripted model
model = torch.jit.load('mnist_cnn_scripted.pt')
model.eval()

# Define a function to preprocess and predict the digit
def predict_digit(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

# Create the Streamlit interface
st.title("MNIST Digit Classifier with TorchScript")

uploaded_file = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        predicted_digit = predict_digit(image)
        st.write(f"Predicted Digit: {predicted_digit}")
