# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

app = Flask(__name__)
CORS(app)  # This is the crucial line that enables the connection!

# This will act as our in-memory database.
# In a production environment, you would use a proper database like Firestore, PostgreSQL, or MongoDB.
VOICEPRINTS_DB = {}
DATABASE_FILE = 'voiceprints.json'

def load_db():
    """Loads voiceprints from a JSON file."""
    if os.path.exists(DATABASE_FILE):
        try:
            with open(DATABASE_FILE, 'r') as f:
                data = json.load(f)
                # Convert list back to numpy array after loading
                for subject, features in data.items():
                    VOICEPRINTS_DB[subject] = np.array(features)
        except json.JSONDecodeError:
            print(f"Warning: {DATABASE_FILE} is corrupted or empty. Starting with an empty database.")
            VOICEPRINTS_DB.clear()
    print("Database loaded:", VOICEPRINTS_DB.keys())

def save_db():
    """Saves voiceprints to a JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_db = {subject: features.tolist() for subject, features in VOICEPRINTS_DB.items()}
    with open(DATABASE_FILE, 'w') as f:
        json.dump(serializable_db, f)

def extract_features(audio_file):
    """
    Extracts Mel-frequency cepstral coefficients (MFCCs) from an audio file.
    This is the core feature extraction step for creating a voiceprint.
    """
    try:
        # Load the audio file, ensuring a consistent sample rate
        y, sr = librosa.load(audio_file, sr=22050)
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Calculate the mean and standard deviation of the MFCCs.
        # This creates a more robust voiceprint.
        mean_mfccs = np.mean(mfccs.T, axis=0)
        std_mfccs = np.std(mfccs.T, axis=0)
        
        # Concatenate mean and standard deviation to create a single feature vector
        voiceprint = np.concatenate((mean_mfccs, std_mfccs))
        
        # Normalize the vector to have a mean of 0 and a standard deviation of 1
        # This is a critical step for robust comparison.
        voiceprint = (voiceprint - np.mean(voiceprint)) / (np.std(voiceprint) + 1e-6) # Added small value to avoid division by zero
        return voiceprint
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# NEW ROUTE: Serves the index.html file directly.
@app.route('/')
def serve_frontend():
    """Serves the main web page by reading the index.html file directly."""
    return send_from_directory('.', 'index.html')


@app.route('/clear_db', methods=['POST'])
def clear_db():
    """Clears all voiceprints from the database."""
    global VOICEPRINTS_DB
    VOICEPRINTS_DB.clear()
    if os.path.exists(DATABASE_FILE):
        os.remove(DATABASE_FILE)
    print("Database cleared.")
    return jsonify({"message": "Database cleared successfully."}), 200

@app.route('/enroll', methods=['POST'])
def enroll_voice():
    """API endpoint to enroll a new subject's voiceprint."""
    subject_name = request.form.get('name')
    audio_file = request.files.get('audio')

    if not subject_name or not audio_file:
        return jsonify({"error": "Missing subject name or audio file."}), 400

    features = extract_features(audio_file)
    if features is None:
        return jsonify({"error": "Could not process audio file."}), 500

    VOICEPRINTS_DB[subject_name] = features
    save_db()
    
    return jsonify({
        "message": f"Successfully enrolled voiceprint for '{subject_name}'.",
        "voiceprint_shape": list(features.shape)
    }), 200

@app.route('/analyze', methods=['POST'])
def analyze_voice():
    """API endpoint to analyze an unknown voice and find a match."""
    audio_file = request.files.get('audio')

    if not audio_file:
        return jsonify({"error": "Missing audio file for analysis."}), 400

    if not VOICEPRINTS_DB:
        return jsonify({"error": "No voiceprints found in the database. Please enroll a subject first."}), 404

    unknown_features = extract_features(audio_file)
    if unknown_features is None:
        return jsonify({"error": "Could not process unknown audio file."}), 500

    results = []
    # Reshape for cosine_similarity
    unknown_features_reshaped = unknown_features.reshape(1, -1)
    
    for subject, known_features in VOICEPRINTS_DB.items():
        known_features_reshaped = known_features.reshape(1, -1)
        similarity = cosine_similarity(unknown_features_reshaped, known_features_reshaped)[0][0]
        results.append({
            "subject": subject,
            "similarity": float(similarity)
        })

    # Sort results by similarity in descending order
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return jsonify({"results": results}), 200

if __name__ == '__main__':
    load_db()
    app.run(debug=True, port=5000)
