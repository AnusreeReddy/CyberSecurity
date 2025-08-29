import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# --- Function to extract MFCC features ---
def extract_features(file):
    y, sr = librosa.load(file, sr=None)
    
    # Plot waveform
    plt.figure(figsize=(8, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform: {file}")
    plt.show()

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Plot MFCCs
    plt.figure(figsize=(8, 3))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title(f"MFCC: {file}")
    plt.show()

    return np.mean(mfcc.T, axis=0)  # average features

# --- Step 3: Build dataset ---
X, y_labels = [], []
data = {
    "fraudster1": ["fraudster1_1.wav", "fraudster1_2.wav"],
    "fraudster2": ["fraudster2_1.wav"]
}

for label, files in data.items():
    for f in files:
        features = extract_features(f)
        X.append(features)
        y_labels.append(label)

X, y_labels = np.array(X), np.array(y_labels)

# --- Step 4: Train Classifier ---
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.25, random_state=42)
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# --- Step 5: Test New Voice ---
test_features = extract_features("test.wav")
prediction = model.predict([test_features])
print("Predicted Speaker:", prediction[0])
 # type: ignore