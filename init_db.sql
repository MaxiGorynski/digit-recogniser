-- Create tables for the application
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    image BYTEA,
    predicted_digit INTEGER,
    confidence REAL,
    true_label INTEGER
);

CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    date DATE,
    accuracy REAL,
    total_predictions INTEGER
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_digits ON predictions(predicted_digit, true_label);

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO alice;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO alice;