import streamlit as st
import streamlit_drawable_canvas as st_canvas
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io

# Import the model definition
from digit_classifier import MNISTClassification


class MNISTPredictor:
    def __init__(self, model_path='mnist_classifier.pth'):
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize and load the model
        self.model = MNISTClassification().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def preprocess_image(self, image):
        # Comprehensive preprocessing with detailed debugging
        try:
            # Ensure grayscale
            if len(image.shape) == 3:
                if image.shape[2] == 3:  # Color image
                    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:  # Multiple channel grayscale
                    img_gray = image[:, :, 0]
            else:
                img_gray = image.copy()

            # Create debugging figure
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Image Preprocessing Steps', fontsize=16)

            # Original image
            axes[0, 0].imshow(img_gray, cmap='gray')
            axes[0, 0].set_title('Original Grayscale')

            # Compute image statistics
            st.sidebar.write("Image Statistics:")
            st.sidebar.write(f"Shape: {img_gray.shape}")
            st.sidebar.write(f"Dtype: {img_gray.dtype}")
            st.sidebar.write(f"Min value: {img_gray.min()}")
            st.sidebar.write(f"Max value: {img_gray.max()}")
            st.sidebar.write(f"Mean value: {img_gray.mean():.2f}")

            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                img_gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            axes[0, 1].imshow(thresh, cmap='gray')
            axes[0, 1].set_title('Adaptive Threshold')

            # Find contours to center the digit
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Find the largest contour (presumably the digit)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Extract and pad the digit
                digit_img = thresh[y:y + h, x:x + w]

                # Show extracted digit
                axes[0, 2].imshow(digit_img, cmap='gray')
                axes[0, 2].set_title('Extracted Digit')

                # Resize with aspect ratio preservation
                aspect = w / h
                if aspect > 1:
                    new_w = 20
                    new_h = int(20 / aspect)
                else:
                    new_h = 20
                    new_w = int(20 * aspect)

                resized = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Create centered 28x28 image
                centered = np.zeros((28, 28), dtype=np.uint8)
                x_offset = (28 - new_w) // 2
                y_offset = (28 - new_h) // 2

                centered[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            else:
                # If no contours, use the whole image
                centered = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)

            # Show centered and resized image
            axes[1, 0].imshow(centered, cmap='gray')
            axes[1, 0].set_title('Centered Digit')

            # Verify digit is white on black
            axes[1, 1].imshow(255 - centered, cmap='gray')
            axes[1, 1].set_title('Inverted for Verification')

            # Prepare for model
            img_for_model = 255 - centered

            # Convert to tensor
            img_tensor = self.transform(img_for_model).unsqueeze(0).to(self.device)

            # Visualize tensor
            tensor_img = img_tensor.squeeze().cpu().numpy()
            axes[1, 2].imshow(tensor_img, cmap='gray')
            axes[1, 2].set_title('Normalized Tensor')

            plt.tight_layout()
            st.pyplot(fig)

            return img_tensor

        except Exception as e:
            st.error(f"Error in image preprocessing: {e}")
            raise

    def predict(self, image):
        # Preprocess the image
        img_tensor = self.preprocess_image(image)

        # Get model prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)

        return {
            'prediction': predicted.item(),
            'confidence': max_prob.item(),
            'probabilities': probabilities.cpu().numpy()[0]
        }


def main():
    st.title('MNIST Digit Recognition - Comprehensive Debug')

    # Initialize the predictor
    predictor = MNISTPredictor()

    # Sidebar for drawing parameters
    st.sidebar.header('Draw a Digit')

    # Create a canvas for drawing
    canvas_result = st_canvas.st_canvas(
        fill_color='#000000',  # Fill color with black
        stroke_width=20,  # Brush size
        stroke_color='#FFFFFF',  # Brush color
        background_color='#000000',  # Background color
        height=280,  # Canvas height
        width=280,  # Canvas width
        drawing_mode='freedraw',  # Draw freehand
        key='canvas'
    )

    # File uploader as an alternative input method
    uploaded_file = st.sidebar.file_uploader(
        "Or upload a digit image",
        type=['png', 'jpg', 'jpeg', 'bmp']
    )

    # Predict button
    predict_button = st.sidebar.button('Predict Digit')

    if predict_button:
        # Determine the image source
        if uploaded_file:
            # Read uploaded image
            image = np.array(Image.open(uploaded_file).convert('L'))
        elif canvas_result.image_data is not None:
            # Use canvas drawing
            image = canvas_result.image_data
        else:
            st.sidebar.error('Please draw a digit or upload an image')
            return

        # Display original image
        st.image(image, caption='Original Image', use_column_width=True)

        try:
            # Run prediction
            result = predictor.predict(image)

            # Display prediction results
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Predicted Digit", result['prediction'])

            with col2:
                st.metric("Confidence", f"{result['confidence']:.2%}")

            # Create probability bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            digits = list(range(10))
            probabilities = result['probabilities']
            ax.bar(digits, probabilities)
            ax.set_xlabel('Digit')
            ax.set_ylabel('Probability')
            ax.set_title('Digit Probabilities')
            ax.set_xticks(digits)
            ax.set_xticklabels(digits)

            # Highlight the predicted digit
            ax.bar(result['prediction'], probabilities[result['prediction']], color='red')

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction error: {e}")


if __name__ == '__main__':
    main()