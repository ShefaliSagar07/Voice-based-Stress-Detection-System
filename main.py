import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# =============================
# SLATE RESEARCH THEME
# =============================
DEEP_SLATE   = "#1e2d3d"   # deep navy slate — titles, borders
MID_SLATE    = "#34495e"   # primary lines / main color
SOFT_SLATE   = "#5d7a8a"   # secondary lines / accent
LIGHT_SLATE  = "#a2b5c1"   # light fills / whiskers
PALE         = "#f0f3f5"   # background / grid
TEXT_COLOR   = "#1a2530"   # axis labels, tick text

# Seaborn / matplotlib base
plt.rcParams.update({
    "figure.facecolor":  PALE,
    "axes.facecolor":    "#f7f9fa",
    "axes.edgecolor":    DEEP_SLATE,
    "axes.labelcolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "grid.color":        "#dce3e8",
    "text.color":        TEXT_COLOR,
    "axes.titlecolor":   DEEP_SLATE,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

SLATE_PALETTE = [LIGHT_SLATE, MID_SLATE]

# =============================
# DATA PATH & SETUP
# =============================
DATA_PATH = r"C:\Users\Hp\Downloads\Voice_Stress_Project\archive (1)"
EMOTIONS = {"01": 0, "05": 1}

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    pitch = librosa.yin(y, fmin=50, fmax=300)
    feature_vector = np.hstack((
        np.mean(mfcc, axis=1),
        zcr,
        spectral_centroid,
        rms
    ))
    return feature_vector, y, sr, mfcc, rms, zcr, spectral_centroid, pitch

X, y_labels = [], []
all_rms, all_zcr, all_centroid = [], [], []
neutral_sample = None
angry_sample   = None

for actor in os.listdir(DATA_PATH):
    actor_path = os.path.join(DATA_PATH, actor)
    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if not file.endswith(".wav"):
                continue
            parts = file.split("-")
            if len(parts) < 3:
                continue
            emotion = parts[2]
            if emotion in EMOTIONS:
                file_path = os.path.join(actor_path, file)
                features, audio, sr, mfcc, rms, zcr, centroid, pitch = extract_features(file_path)
                X.append(features)
                y_labels.append(EMOTIONS[emotion])
                all_rms.append(rms)
                all_zcr.append(zcr)
                all_centroid.append(centroid)
                if EMOTIONS[emotion] == 0 and neutral_sample is None:
                    neutral_sample = (audio, sr, pitch)
                if EMOTIONS[emotion] == 1 and angry_sample is None:
                    angry_sample = (audio, sr, pitch)

X        = np.array(X)
y_labels = np.array(y_labels)
print("Total samples:", len(X))

# =============================
# MODEL
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_labels, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =============================
# 1️⃣  Waveform
# =============================
audio, sr, _ = neutral_sample

fig, ax = plt.subplots(figsize=(10, 4))
librosa.display.waveshow(audio, sr=sr, color=MID_SLATE, ax=ax, alpha=0.85)
ax.set_title("Speech Waveform (Neutral Sample)", fontsize=13, fontweight="bold")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude")
fig.tight_layout()
fig.savefig("1_waveform.png", dpi=150)
plt.close()

# =============================
# 2️⃣  MFCC Trajectory
# =============================
mfcc_data  = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
mfcc_colors = [MID_SLATE, SOFT_SLATE, LIGHT_SLATE]
mfcc_labels = ["MFCC 1", "MFCC 2", "MFCC 3"]

fig, ax = plt.subplots(figsize=(10, 5))
for i in range(3):
    ax.plot(mfcc_data[i], color=mfcc_colors[i], linewidth=1.6, label=mfcc_labels[i])
ax.set_title("MFCC Coefficients Over Time", fontsize=13, fontweight="bold")
ax.set_xlabel("Frame Index")
ax.set_ylabel("Coefficient Value")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("2_mfcc.png", dpi=150)
plt.close()

# =============================
# 3️⃣  Rolling Variance of Energy
# =============================
rms_full = librosa.feature.rms(y=audio)[0]
rolling  = [np.var(rms_full[i:i+10]) for i in range(len(rms_full)-10)]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(rolling, color=MID_SLATE, linewidth=1.8)
ax.fill_between(range(len(rolling)), rolling, alpha=0.18, color=SOFT_SLATE)
ax.set_title("Rolling Variance of Energy", fontsize=13, fontweight="bold")
ax.set_xlabel("Frame Index")
ax.set_ylabel("Variance")
fig.tight_layout()
fig.savefig("3_rolling_variance.png", dpi=150)
plt.close()

# =============================
# 4️⃣  Energy (RMS) Comparison
# =============================
neutral_rms = [all_rms[i] for i in range(len(y_labels)) if y_labels[i] == 0]
angry_rms   = [all_rms[i] for i in range(len(y_labels)) if y_labels[i] == 1]

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=[neutral_rms, angry_rms],
            palette=SLATE_PALETTE,
            linewidth=1.5,
            flierprops=dict(marker='o', color=DEEP_SLATE, markersize=4),
            ax=ax)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Neutral", "Angry"])
ax.set_title("Energy Comparison: Neutral vs Angry", fontsize=13, fontweight="bold")
ax.set_ylabel("Average RMS Energy")
fig.tight_layout()
fig.savefig("4_energy_comparison.png", dpi=150)
plt.close()

# =============================
# 5️⃣  Confusion Matrix
# =============================
cm = confusion_matrix(y_test, y_pred)
slate_cmap = sns.light_palette(MID_SLATE, as_cmap=True)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d',
            cmap=slate_cmap,
            linewidths=0.5,
            linecolor=PALE,
            xticklabels=["Neutral", "Angry"],
            yticklabels=["Neutral", "Angry"],
            ax=ax)
ax.set_title("Confusion Matrix (Raw Counts)", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
fig.tight_layout()
fig.savefig("5_confusion_matrix.png", dpi=150)
plt.close()

# =============================
# 6️⃣  Mel Spectrogram Side-by-Side
# =============================
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.patch.set_facecolor(PALE)

for idx, (sample, label) in enumerate([(neutral_sample, "Neutral"), (angry_sample, "Angry")]):
    mel = librosa.feature.melspectrogram(y=sample[0], sr=sample[1], n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sample[1], x_axis='time', y_axis='mel',
                                   ax=axes[idx], cmap="Blues")
    axes[idx].set_title(f"Mel Spectrogram — {label}", fontsize=12, fontweight="bold", color=DEEP_SLATE)
    axes[idx].set_xlabel("Time (s)", color=TEXT_COLOR)
    axes[idx].set_ylabel("Hz" if idx == 0 else "", color=TEXT_COLOR)
    fig.colorbar(img, ax=axes[idx], format="%+2.0f dB")

fig.tight_layout()
fig.savefig("6_mel_spectrogram_comparison.png", dpi=150)
plt.close()

# =============================
# 7️⃣  Pitch Contour Comparison
# =============================
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(neutral_sample[2], color=SOFT_SLATE,  linewidth=1.6, label="Neutral", alpha=0.85)
ax.plot(angry_sample[2],   color=DEEP_SLATE, linewidth=1.6, label="Angry",   alpha=0.85)
ax.set_title("Pitch Contour (F0) Comparison", fontsize=13, fontweight="bold")
ax.set_xlabel("Frame Index")
ax.set_ylabel("Frequency (Hz)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("7_pitch_contour.png", dpi=150)
plt.close()

# =============================
# 8️⃣  ZCR Comparison
# =============================
neutral_zcr = [all_zcr[i] for i in range(len(y_labels)) if y_labels[i] == 0]
angry_zcr   = [all_zcr[i] for i in range(len(y_labels)) if y_labels[i] == 1]

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=[neutral_zcr, angry_zcr],
            palette=SLATE_PALETTE,
            linewidth=1.5,
            flierprops=dict(marker='o', color=DEEP_SLATE, markersize=4),
            ax=ax)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Neutral ZCR", "Angry ZCR"])
ax.set_title("Zero Crossing Rate Comparison", fontsize=13, fontweight="bold")
ax.set_ylabel("Zero Crossing Rate")
fig.tight_layout()
fig.savefig("8_zcr_comparison.png", dpi=150)
plt.close()

print("\nAll 8 slate-themed research visuals generated successfully.")