import psycopg2
from psycopg2 import pool
import pandas as pd
from datetime import datetime


class DatabaseManager:
    def __init__(self, host="localhost", database="digit_recognizer",
                 user="alice", password="inwonderland", min_conn=1, max_conn=5):
        """Initialize the database connection pool"""
        self.connection_params = {
            "host": host,
            "database": database,
            "user": user,
            "password": password
        }
        self.min_conn = min_conn
        self.max_conn = max_conn
        self.pool = None
        self.conn = None  # Single connection instead of pool

    def initialize(self):
        """Create a single connection instead of a pool"""
        if self.conn is None:
            try:
                # Create a single connection instead of a pool
                self.conn = psycopg2.connect(**self.connection_params)

                # Create tables
                self._create_tables()
                return True
            except Exception as e:
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
            self.conn.commit()
            cursor.close()
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            print(f"Error creating tables: {e}")

    def log_prediction(self, image_bytes, prediction, confidence, true_label):
        """Log a prediction to the database"""
        if self.conn is None:
            if not self.initialize():
                return False

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO predictions (timestamp, image, predicted_digit, confidence, true_label) VALUES (%s, %s, %s, %s, %s)",
                (datetime.now(), psycopg2.Binary(image_bytes), prediction, confidence, true_label)
            )
            self.conn.commit()
            cursor.close()
            return True
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            print(f"Error logging prediction: {e}")
            return False

    def get_recent_predictions(self, limit=100):
        """Get recent predictions from the database"""
        if self.conn is None:
            if not self.initialize():
                return None

        try:
            query = """
                SELECT id, timestamp, predicted_digit, confidence, true_label 
                FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT %s
            """
            df = pd.read_sql_query(query, self.conn, params=(limit,))
            return df
        except Exception as e:
            print(f"Error fetching predictions: {e}")
            return None

    def get_model_accuracy(self):
        """Calculate model accuracy from stored predictions"""
        if self.conn is None:
            if not self.initialize():
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
            print(f"Error calculating accuracy: {e}")
            return None, 0

    def close(self):
        """Close the connection"""
        if self.conn:
            self.conn.close()
            self.conn = None