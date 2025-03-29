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

# Import model and db utils
from digit_classifier import MNISTNet
from postgres_db_util import DatabaseManager

# Init db manager
db_manager = DatabaseManager(
    host='localhost',  # IMPORTANT: Use localhost NOT postgres for local testing
    database='digit_recognizer',
    user='alice',
    password='inwonderland'
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
    # Convert to greyscale
    if image.mode != 'L':
        image = image.convert('L')

    # Resize to 28x28
    image = image.resize((28, 28))

    # Apply same normalization as training data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return transform(image).unsqueeze(0)  # Add batch dim


# Make a prediction
def predict_digit(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        pred_prob, pred_class = torch.max(probabilities, 1)

        # Get all class probs
        all_probs = probabilities.squeeze().numpy()

        # Special handling for 6 vs 9 confusion
        pred_digit = pred_class.item()
        if pred_digit == 9:
            # Double-check if it might be a 6 instead
            # Simple heuristic: check if there's more "mass" in the upper part of the image
            # Since 6 has its loop at the bottom, while 9 has its loop at the top
            img = image_tensor.squeeze().numpy()

            # Split image into top and bottom halves
            top_half = img[:14, :]  # Upper half
            bottom_half = img[14:, :]  # Lower half

            # Calculate "mass" (sum of pixel values)
            top_mass = np.sum(np.abs(top_half))
            bottom_mass = np.sum(np.abs(bottom_half))

            # Check if most of the "ink" is in the bottom half
            if bottom_mass > top_mass * 1.2:
                # More likely to be a 6
                pred_digit = 6
                pred_confidence = pred_prob.item() * 0.9

                # Update probs for visualization
                all_probs = all_probs.copy()
                all_probs[9] = all_probs[9] * 0.5
                all_probs[6] = max(all_probs[6] + all_probs[9], 0.8)

                return pred_digit, pred_confidence, all_probs

        return pred_digit, pred_prob.item(), all_probs


def main():
    st.set_page_config(
        page_title="MNIST Digit Recognizer",
        page_icon="🔢",
        layout="wide"
    )

    st.title("Handwritten Digit Recognizer")

    # Initialize db
    if not db_manager.initialize():
        st.error("Failed to connect to database. Check ur connection settings.")

    # Navigation
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Draw & Predict", "Model Performance", "Recent Predictions"]
    )

    # Load model
    model = load_model()
    if model is None:
        st.error("Model file 'mnist_model.pth' not found. Plz train the model first.")

        if st.button("Train New Model"):
            st.info("Training new model... This may take a few mins.")
            with st.spinner("Training in progress..."):
                from mnist_model import train_model
                model, accuracy = train_model(epochs=5, focus_digits=[6])
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

                # Display pred and confidence
                st.markdown(f"## Prediction: **{prediction}**")
                st.markdown(f"Confidence: {confidence * 100:.2f}%")

                # Plot probs
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.bar(range(10), all_probs, color='skyblue')
                bars[prediction].set_color('navy')
                ax.set_xticks(range(10))
                ax.set_xlabel('Digit')
                ax.set_ylabel('Probability')
                ax.set_title('Prediction Probabilities')
                st.pyplot(fig)

                # Save image to bytes for db
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
                    # Log to db
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

            # Special analysis for 6 vs 9 confusion
            st.subheader("6 vs 9 Analysis")

            # Filter for instances where true label is 6 or 9
            df_6_9 = df[(df['true_label'] == 6) | (df['true_label'] == 9)]

            if not df_6_9.empty:
                # Calculate accuracy for 6 and 9
                accuracy_6 = df_6_9[df_6_9['true_label'] == 6]['correct'].mean()
                accuracy_9 = df_6_9[df_6_9['true_label'] == 9]['correct'].mean()

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Accuracy for digit 6", f"{accuracy_6 * 100:.2f}%" if not pd.isna(accuracy_6) else "N/A")

                with col2:
                    st.metric("Accuracy for digit 9", f"{accuracy_9 * 100:.2f}%" if not pd.isna(accuracy_9) else "N/A")

                # Create 2x2 confusion matrix for 6 and 9
                df_6_9_copy = df_6_9.copy()
                # Map true labels and predictions to 0/1 (6/9)
                df_6_9_copy['true_binary'] = df_6_9_copy['true_label'].apply(lambda x: 0 if x == 6 else 1)
                df_6_9_copy['pred_binary'] = df_6_9_copy['predicted_digit'].apply(lambda x: 0 if x == 6 else 1)

                # Create the binary confusion matrix
                conf_6_9 = pd.crosstab(
                    df_6_9_copy['true_binary'],
                    df_6_9_copy['pred_binary'],
                    rownames=['True'],
                    colnames=['Predicted'],
                    normalize='index'
                )

                # Plot 6 vs 9 confusion matrix
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                im2 = ax2.imshow(conf_6_9, cmap='Blues')

                # Add text annotations
                for i in range(len(conf_6_9.index)):
                    for j in range(len(conf_6_9.columns)):
                        if not pd.isna(conf_6_9.iloc[i, j]):
                            text = ax2.text(j, i, f"{conf_6_9.iloc[i, j]:.2f}",
                                            ha="center", va="center",
                                            color="white" if conf_6_9.iloc[i, j] > 0.5 else "black")

                # Set ticks
                ax2.set_xticks([0, 1])
                ax2.set_yticks([0, 1])

                # Label ticks
                ax2.set_xticklabels(['6', '9'])
                ax2.set_yticklabels(['6', '9'])

                # Add title
                ax2.set_title("6 vs 9 Confusion Matrix")
                ax2.set_xlabel("Predicted Label")
                ax2.set_ylabel("True Label")

                st.pyplot(fig2)

                st.info(
                    "If issues with 6/9 confusion persist, consider retraining the model with more examples of these digits.")
            else:
                st.info("No data available yet for digits 6 and 9.")
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