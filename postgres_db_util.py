import psycopg2
import pandas as pd
from datetime import datetime
import streamlit as st
import os


class DatabaseManager:
    def __init__(self, host=None, database=None, user=None, password=None):
        """Initialize the database connection"""
        # Use environment variables if provided, otherwise use defaults
        self.connection_params = {
            "host": host or os.environ.get('DB_HOST', 'localhost'),
            "database": database or os.environ.get('DB_NAME', 'digit_recognizer'),
            "user": user or os.environ.get('DB_USER', 'alice'),
            "password": password or os.environ.get('DB_PASSWORD', 'inwonderland')
        }
        self.conn = None
        print(f"Database connection params (host): {self.connection_params['host']}")

    def initialize(self):
        """Create a database connection"""
        if self.conn is None:
            try:
                # Create a single connection
                self.conn = psycopg2.connect(**self.connection_params)

                # Create tables
                self._create_tables()
                print("Database connection successful")
                return True
            except Exception as e:
                st.error(f"Database connection error: {e}")
                print(f"Error initializing database: {e}")
                return False
        return True

    def _create_tables(self):
        """Create the necessary tables if they don't exist"""
        try:
            cursor = self.conn.cursor()
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
                    image BYTEA,
                    predicted_digit INTEGER,
                    confidence REAL,
                    true_label INTEGER
                )
            ''')

            # Model performance summary table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id SERIAL PRIMARY KEY,
                    date DATE,
                    accuracy REAL,
                    total_predictions INTEGER
                )
            ''')

            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)')
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_predictions_digits ON predictions(predicted_digit, true_label)')

            self.conn.commit()
            cursor.close()
            print("Database tables created successfully")
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            st.error(f"Table creation error: {e}")
            print(f"Error creating tables: {e}")

    def log_prediction(self, image_bytes, prediction, confidence, true_label):
        """Log a prediction to the database"""
        if self.conn is None:
            if not self.initialize():
                st.error("Failed to initialize database connection")
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
            st.error(f"Error logging prediction: {e}")
            print(f"Error logging prediction: {e}")
            return False

    def get_recent_predictions(self, limit=100):
        """Get recent predictions from the database"""
        if self.conn is None:
            if not self.initialize():
                st.error("Database connection not available")
                return None

        try:
            query = """
                SELECT id, timestamp, predicted_digit, confidence, true_label 
                FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT %s
            """
            df = pd.read_sql_query(query, self.conn, params=(limit,))
            print(f"Retrieved {len(df)} predictions from database")
            return df
        except Exception as e:
            st.error(f"Error fetching predictions: {e}")
            print(f"Error fetching predictions: {e}")
            return None

    def get_model_accuracy(self):
        """Calculate model accuracy from stored predictions"""
        if self.conn is None:
            if not self.initialize():
                st.error("Database connection not available")
                return None, 0

        try:
            query = """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END) as correct
                FROM predictions
                WHERE true_label IS NOT NULL
            """
            cursor = self.conn.cursor()
            cursor.execute(query)
            total, correct = cursor.fetchone()
            cursor.close()

            if total > 0:
                accuracy = correct / total
                return accuracy, total
            return 0, 0
        except Exception as e:
            st.error(f"Error calculating accuracy: {e}")
            print(f"Error calculating accuracy: {e}")
            return None, 0

    def close(self):
        """Close the connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("Database connection closed")