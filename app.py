from flask import Flask, request, jsonify, render_template
import sqlite3
from datetime import datetime
from flask_cors import CORS
import matplotlib.pyplot as plt
import os
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# SQLite DB setup
DATABASE = 'sensor_data.db'

# Create DB connection
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize DB with extended fields
def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sensor_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temperature REAL,
            humidity REAL,
            vibration REAL,
            current REAL,
            voltage REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

clf = joblib.load("clf.joblib")
reg = joblib.load("regressor.joblib")

# Route to serve HTML
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to receive sensor data
@app.route('/sensor', methods=['GET'])
def receive_sensor_data():
    try:
        temperature = float(request.args.get('temperature', 0))
        humidity = float(request.args.get('humidity', 0))
        vibration = float(request.args.get('vibration', 0))
        current = float(request.args.get('current', 0))
        voltage = float(request.args.get('voltage', 0))

        conn = get_db_connection()
        conn.execute('''
            INSERT INTO sensor_values (temperature, humidity, vibration, current, voltage)
            VALUES (?, ?, ?, ?, ?)
        ''', (temperature, humidity, vibration, current, voltage))
        conn.commit()
        conn.close()

        return "Sensor values stored successfully", 200
    except Exception as e:
        app.logger.error(f"Error storing sensor data: {e}")
        return "Invalid input or database error", 500

# Endpoint to fetch sensor history as JSON
@app.route('/history', methods=['GET'])
def get_history():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM sensor_values ORDER BY timestamp DESC LIMIT 100").fetchall()
    history = [dict(row) for row in rows]
    conn.close()
    return jsonify(history)

# Endpoint to render and save graphs
@app.route('/graphs')
def generate_graphs():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM sensor_values ORDER BY timestamp").fetchall()
    conn.close()

    if not rows:
        return "No data to plot."

    timestamps = [row['timestamp'] for row in rows]
    temp = [row['temperature'] for row in rows]
    hum = [row['humidity'] for row in rows]
    vib = [row['vibration'] for row in rows]
    cur = [row['current'] for row in rows]
    volt = [row['voltage'] for row in rows]

    os.makedirs('static', exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, temp, label='Temperature (°C)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig('static/temp.png')
    plt.clf()

    plt.plot(timestamps, hum, label='Humidity (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig('static/humidity.png')
    plt.clf()

    plt.plot(timestamps, vib, label='Vibration (V)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig('static/vibration.png')
    plt.clf()

    plt.plot(timestamps, cur, label='Current (A)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig('static/current.png')
    plt.clf()

    plt.plot(timestamps, volt, label='Voltage (V)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig('static/voltage.png')
    plt.clf()

    return render_template('graphs.html')

# Predict using latest DB values
@app.route('/predict', methods=['GET'])
def predict():
    try:
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM sensor_values ORDER BY timestamp DESC LIMIT 1").fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "No data found"}), 404

        features = [
            row['temperature'],
            row['humidity'],  # treated as pressure
            row['vibration'],
            row['voltage'],
            row['current']
        ]

        df = pd.DataFrame([features], columns=['temperature', 'pressure', 'vibration', 'voltage', 'current'])

        anomaly = int(clf.predict(df)[0])
        time_left = float(reg.predict(df)[0])

        return jsonify({
            "anomaly": anomaly,
            "time_to_failure": time_left
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict_ui')
def prediction_ui():
    return render_template("predict.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)