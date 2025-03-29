import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import pandas as pd
import torchvision.transforms as transforms
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

# Import our model class and database utilities
from mnist_model import MNISTNet
from db_utils import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager(
    host=os.getenv('DB_HOST', 'localhost'),
    database=os.getenv('DB_NAME', 'digit_recognizer'),
    user=os.getenv('DB_USER', 'alice'),
    password=os.getenv('DB_PASSWORD', 'inwonderland')
)


# Load the trained model
@st.cache_resource
def load_model():
    model = MNISTNet()
    try:
        model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        return None


# Preprocess the image for the model
def preprocess_image(image):
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')

    # Resize to 28x28
    image = image.resize((28, 28))

    # Apply same normalization as training data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return transform(image).unsqueeze(0)  # Add batch dimension


# Make a prediction
def predict_digit(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        pred_prob, pred_class = torch.max(probabilities, 1)

        # Get all class probabilities
        all_probs = probabilities.squeeze().numpy()

        return pred_class.item(), pred_prob.item(), all_probs


def main():
    st.set_page_config(
        page_title="MNIST Digit Recognizer",
        page_icon="🔢",
        layout="wide"
    )

    st.title("Handwritten Digit Recognizer")

    # Initialize database
    if not db_manager.initialize():
        st.error("Failed to connect to database. Check your connection settings.")

    # Navigation
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Draw & Predict", "Model Performance", "Recent Predictions"]
    )

    # Load model
    model = load_model()
    if model is None:
        st.error("Model file 'mnist_model.pth' not found. Please train the model first.")

        if st.button("Train New Model"):
            st.info("Training new model... This may take a few minutes.")
            with st.spinner("Training in progress..."):
                from mnist_model import train_model
                model, accuracy = train_model(epochs=5)
                st.success(f"Model trained successfully with {accuracy:.2f}% accuracy!")
                st.experimental_rerun()

        return

    # Pages
    if page == "Draw & Predict":
        draw_predict_page(model)
    elif page == "Model Performance":
        model_performance_page()
    elif page == "Recent Predictions":
        recent_predictions_page()


def draw_predict_page(model):
    st.header("Draw a digit (0-9)")

    col1, col2 = st.columns([3, 2])

    with col1:
        # Create canvas for drawing
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

    prediction = None
    confidence = None
    all_probs = None

    if canvas_result.image_data is not None:
        # Convert canvas to PIL Image
        img_array = canvas_result.image_data.astype(np.uint8)
        if img_array.sum() > 0:  # Check if canvas is not empty
            image = Image.fromarray(img_array).convert('L')

            # Preprocess and predict
            image_tensor = preprocess_image(image)

            with col2:
                st.write("Preprocessed for model (28x28):")
                processed_img = Image.fromarray(
                    (image_tensor.squeeze().numpy() * 0.3081 + 0.1307) * 255
                ).convert('L')
                st.image(processed_img, width=150)

            if st.button("Predict"):
                prediction, confidence, all_probs = predict_digit(model, image_tensor)

                # Display prediction and confidence
                st.markdown(f"## Prediction: **{prediction}**")
                st.markdown(f"Confidence: {confidence * 100:.2f}%")

                # Plot probabilities
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.bar(range(10), all_probs, color='skyblue')
                bars[prediction].set_color('navy')
                ax.set_xticks(range(10))
                ax.set_xlabel('Digit')
                ax.set_ylabel('Probability')
                ax.set_title('Prediction Probabilities')
                st.pyplot(fig)

                # Save image to bytes for database
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()

                # Get true label from user
                true_label = st.number_input(
                    "If you'd like to provide feedback, what was the true digit?",
                    min_value=0,
                    max_value=9,
                    value=prediction,
                    step=1
                )

                if st.button("Submit Feedback"):
                    # Log to database
                    if db_manager.log_prediction(img_bytes, prediction, confidence, true_label):
                        st.success(f"Logged prediction: {prediction} (true: {true_label})")
                    else:
                        st.error("Failed to log prediction to database")


def model_performance_page():
    st.header("Model Performance")

    # Get accuracy from database
    accuracy, total = db_manager.get_model_accuracy()

    if accuracy is not None:
        st.metric("Overall Accuracy", f"{accuracy * 100:.2f}%")
        st.metric("Total Predictions", total)

        # Get confusion matrix data
        df = db_manager.get_recent_predictions(limit=1000)

        if df is not None and not df.empty:
            # Create confusion matrix
            conf_matrix = pd.crosstab(
                df['true_label'],
                df['predicted_digit'],
                rownames=['True'],
                colnames=['Predicted'],
                normalize='index'
            )

            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(conf_matrix, cmap='Blues')

            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)

            # Set ticks
            ax.set_xticks(np.arange(10))
            ax.set_yticks(np.arange(10))

            # Label ticks
            ax.set_xticklabels(range(10))
            ax.set_yticklabels(range(10))

            # Add title
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

            # Loop over data dimensions and create text annotations
            for i in range(10):
                for j in range(10):
                    if not np.isnan(conf_matrix.iloc[i, j]):
                        text = ax.text(j, i, f"{conf_matrix.iloc[i, j]:.2f}",
                                       ha="center", va="center",
                                       color="white" if conf_matrix.iloc[i, j] > 0.5 else "black")

            st.pyplot(fig)
        else:
            st.info("Not enough data to create a confusion matrix")
    else:
        st.info("No prediction data available yet")


def recent_predictions_page():
    st.header("Recent Predictions")

    # Get recent predictions
    df = db_manager.get_recent_predictions(limit=100)

    if df is not None and not df.empty:
        # Calculate accuracy
        df['correct'] = df['predicted_digit'] == df['true_label']

        # Format confidence as percentage
        df['confidence'] = df['confidence'].apply(lambda x: f"{x * 100:.2f}%")

        # Format timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Display as table
        st.dataframe(
            df[['timestamp', 'predicted_digit', 'true_label', 'confidence', 'correct']],
            use_container_width=True
        )

        # Display accuracy metrics
        col1, col2 = st.columns(2)
        with col1:
            accuracy = df['correct'].mean()
            st.metric("Recent Accuracy", f"{accuracy * 100:.2f}%")

        with col2:
            st.metric("Total Recent Predictions", len(df))

        # Add a download button for the data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download data as CSV",
            csv,
            "mnist_predictions.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.info("No prediction data available yet")


if __name__ == "__main__":
    main()

# Cleanup database connection on exit
import atexit

atexit.register(db_manager.close)