import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
from datetime import datetime
import psycopg2
from psycopg2 import pool
import torchvision.transforms as transforms
from streamlit_drawable_canvas import st_canvas

#Model import
from digit_classifier import MNISTNet

#Db connection pool
connection_pool = None


def initialize_db_pool():
    global connection_pool
    # Update these with your PostgreSQL credentials
    connection_pool = psycopg2.pool.SimpleConnectionPool(
        1, 10,
        host="postgres",  # Docker service name
        database="digit_recognizer",
        user="postgres",
        password="postgres"
    )

    #Create table if none
    conn = connection_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
                    image BYTEA,
                    predicted_digit INTEGER,
                    confidence REAL,
                    true_label INTEGER
                )
            ''')
        conn.commit()
    except Exception as e:
        st.error(f"Database error: {e}")
    finally:
        connection_pool.putconn(conn)


#Prediction to PostgreSQL
def log_prediction(self, image_bytes, prediction, confidence, true_label):
    """Log a prediction to the database"""
    if self.conn is None:
        if not self.initialize():
            print("Failed to initialize database connection")
            return False

    try:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (timestamp, image, predicted_digit, confidence, true_label) VALUES (%s, %s, %s, %s, %s)",
            (datetime.now(), psycopg2.Binary(image_bytes), prediction, confidence, true_label)
        )
        self.conn.commit()
        cursor.close()
        print(f"Successfully logged prediction: {prediction} (true: {true_label})")
        return True
    except Exception as e:
        if self.conn:
            self.conn.rollback()
        print(f"Error logging prediction: {e}")
        return False


#Load trained model
@st.cache_resource
def load_model():
    model = MNISTNet()
    model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model


#Preprocessing
def preprocess_image(image):
    #Conv. greyscale
    if image.mode != 'L':
        image = image.convert('L')

    #Resize
    image = image.resize((28, 28))

    #Normalise identically to training data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return transform(image).unsqueeze(0)  # Add batch dimension


#Prediction step
def predict_digit(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        pred_prob, pred_class = torch.max(probabilities, 1)
        return pred_class.item(), pred_prob.item()


def main():
    st.title("Handwritten Digit Recognizer")
    st.write("Draw a digit (0-9) in the canvas below")

    #Init. db connection
    if connection_pool is None:
        try:
            initialize_db_pool()
        except Exception as e:
            st.error(f"Failed to connect to database: {e}")

    #Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}. Make sure you've trained the model first.")
        return

    #Build canvas for drawing
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

    col1, col2 = st.columns(2)

    prediction = None
    confidence = None

    if canvas_result.image_data is not None:
        #Canvas to PIL image
        img_array = canvas_result.image_data.astype(np.uint8)
        if img_array.sum() > 0:  # Check if canvas is not empty
            image = Image.fromarray(img_array).convert('L')

            with col1:
                st.write("Your drawing:")
                st.image(image, width=150)

            #Preprocessing/predictio
            image_tensor = preprocess_image(image)

            with col2:
                st.write("Preprocessed for model:")
                processed_img = Image.fromarray(
                    (image_tensor.squeeze().numpy() * 255).astype(np.uint8)
                )
                st.image(processed_img, width=150)

            if st.button("Predict"):
                prediction, confidence, all_probs = predict_digit(model, image_tensor)

                # Store in session state
                st.session_state.prediction = prediction
                st.session_state.confidence = confidence
                st.session_state.all_probs = all_probs
                st.session_state.img_bytes = img_byte_arr.getvalue()

                # Display prediction and confidence
                st.markdown(f"## Prediction: **{prediction}**")
                st.markdown(f"Confidence: {confidence * 100:.2f}%")

                #Image to bytes for saving in db
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()

                #Prompt true label input from user
                true_label = st.number_input("What was the true digit?",
                                             min_value=0,
                                             max_value=9,
                                             value=prediction,
                                             step=1)

                if st.button("Submit Feedback"):
                    # Check if prediction exists in session state
                    if "prediction" in st.session_state:
                        # Log to db
                        if db_manager.log_prediction(
                                st.session_state.img_bytes,
                                st.session_state.prediction,
                                st.session_state.confidence,
                                true_label
                        ):
                            st.success(f"Logged prediction: {st.session_state.prediction} (true: {true_label})")
                        else:
                            st.error("Failed to log prediction to database")


if __name__ == "__main__":
    main()