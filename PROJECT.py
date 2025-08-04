import streamlit as st
import torch
import torch.nn as nn
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns

# --- Configuration ---
BIRD_IMAGES_DIR = r"C:\\Users\\Rabinson\\AppData\\Local\\Programs\\Python\\Python311\\Birdsmar30\\bird_images"
CHECKPOINT_PATH = r"C:\\Users\\Rabinson\\AppData\\Local\\Programs\\Python\\Python311\\Birdsmar30\\best_bird_model.pth"
SEGMENT_DURATION_SINGLE = 10  # seconds
SEGMENT_DURATION_MULTI = 20  # seconds
HOP_FACTOR = 2  # 50% overlap

# --- Load Model ---
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = torch.hub.load('pytorch/vision', 'efficientnet_b2', pretrained=False)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, len(checkpoint["classes"]))
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    return model, checkpoint["class_to_idx"], checkpoint["transform"], device

# --- Audio to Log-Mel Spectrogram ---
def audio_to_log_mel(y, sr, save_path="temp_spectrogram.png", for_prediction=True):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec + 1e-6, ref=np.max)

    plt.figure(figsize=(6, 3))

    if for_prediction:
        librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, cmap='magma')
        plt.axis("off")
    else:
        librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', cmap='magma')
        plt.colorbar(format='%+2.0f dB', label='dB')
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return save_path

# --- Predict Bird Species ---
def predict_audio_segment(model, transform, device, y, sr, idx_to_class):
    spec_path = audio_to_log_mel(y, sr, save_path="temp_spectrogram.png", for_prediction=True)
    img = Image.open(spec_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probabilities, 1)

    species = idx_to_class[pred.item()].replace("_audio", "")
    return species, confidence.item() * 100

# --- Find Bird Image ---
def get_bird_image(species):
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(BIRD_IMAGES_DIR, species + ext)
        if os.path.exists(path):
            return path
    return None

# --- UI Setup ---
st.set_page_config(page_title="Avian Echo – Bird Recognition and Counting App", layout="wide")

st.markdown("""
    <h1 style='text-align: center;'>🕊️ Avian Echo</h1>
    <h4 style='text-align: center;'>A Bird Recognition and Counting App</h4>
    <hr>
""", unsafe_allow_html=True)

st.markdown("Upload an audio file to classify bird species using a deep learning model.")
mode = st.radio("Choose Mode:", ["Single Bird", "Multiple Birds"], horizontal=True)

# --- Load Model Once ---
model, class_to_idx, transform, device = load_model(CHECKPOINT_PATH)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# --- Audio Upload ---
uploaded_audio = st.file_uploader("🎵 Upload an audio file (MP3/WAV)", type=["mp3", "wav"])

if uploaded_audio:
    audio_path = "temp_audio." + uploaded_audio.name.split(".")[-1]
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.read())

    y, sr = librosa.load(audio_path, sr=22050)
    total_duration = len(y) / sr
    st.success(f"⏱ Audio Duration: **{total_duration:.2f} seconds**")

    # ✅ Audio playback
    st.audio(audio_path, format='audio/wav')

    if mode == "Single Bird":
        with st.spinner("Analyzing..."):
            best_conf, best_species, best_segment = 0, "Unknown", None
            for start in range(0, len(y) - sr * SEGMENT_DURATION_SINGLE, sr):
                seg = y[start:start + sr * SEGMENT_DURATION_SINGLE]
                species, conf = predict_audio_segment(model, transform, device, seg, sr, idx_to_class)
                if conf > best_conf:
                    best_conf, best_species, best_segment = conf, species, seg

        audio_to_log_mel(best_segment, sr, "best_spectrogram.png", for_prediction=False)
        bird_img = get_bird_image(best_species)

        col1, col2, col3 = st.columns([1.5, 1, 1.5])
        with col1:
            st.image("best_spectrogram.png", caption="🎼 Log-Mel Spectrogram", use_container_width=True)
        with col2:
            st.markdown(f"### 🔍 Predicted:\n**{best_species}**")
            st.markdown(f"### 🧠 Confidence:\n**{best_conf:.2f}%**")
        with col3:
            if bird_img:
                st.image(bird_img, caption=best_species, use_container_width=True)
            else:
                st.warning("⚠ No image available for this bird.")

    elif mode == "Multiple Birds":
        st.subheader("🔎 Segment-wise Predictions")
        species_log, species_count, true_labels, pred_labels = [], defaultdict(int), [], []

        with st.spinner("Segmenting and predicting..."):
            hop = sr * SEGMENT_DURATION_MULTI // HOP_FACTOR
            for start in range(0, len(y) - sr * SEGMENT_DURATION_MULTI, hop):
                seg = y[start:start + sr * SEGMENT_DURATION_MULTI]
                species, conf = predict_audio_segment(model, transform, device, seg, sr, idx_to_class)
                species_log.append((start/sr, (start+sr*SEGMENT_DURATION_MULTI)/sr, species, conf))
                species_count[species] += 1
                pred_labels.append(species)
                true_labels.append(species)  # Placeholder; replace if real labels exist

                with st.expander(f"{start/sr:.1f}s - {(start+sr*SEGMENT_DURATION_MULTI)/sr:.1f}s → {species} ({conf:.2f}%)"):
                    audio_to_log_mel(seg, sr, "segment_spectrogram.png", for_prediction=False)
                    bird_img = get_bird_image(species)

                    col1, col2, col3 = st.columns([1.5, 1, 1.5])
                    with col1:
                        st.image("segment_spectrogram.png", caption=f"🎼 Spectrogram of {species}", use_container_width=True)
                    with col2:
                        st.markdown(f"### 🔍\n**{species}**")
                        st.markdown(f"### 🧠\n**{conf:.2f}%**")
                    with col3:
                        if bird_img:
                            st.image(bird_img, caption=species, use_container_width=True)
                        else:
                            st.warning("⚠ No image available for this bird.")

        # --- Summary Section ---
        st.divider()
        st.subheader("📊 Detection Summary")

        summary_mode = st.selectbox("Choose summary type", [
            "📜 Full Bird Prediction Log",
            "📈 Most Frequent Bird Detected",
            "🦜 Unique Bird Species Detected",
            "📉 Confusion Matrix"
        ])

        if summary_mode == "📜 Full Bird Prediction Log":
            st.write("### Time-wise Bird Predictions:")
            for start, end, species, conf in species_log:
                st.write(f"**{species}** from `{start:.1f}s` to `{end:.1f}s` — {conf:.2f}%")

        elif summary_mode == "📈 Most Frequent Bird Detected":
            if species_count:
                common = Counter(species_count).most_common(1)[0]
                st.success(f"🥇 **{common[0]}** was detected **{common[1]} times**")
            else:
                st.info("No detections to summarize.")

        elif summary_mode == "🦜 Unique Bird Species Detected":
            st.write(f"🔢 **{len(species_count)} unique species** detected:")
            for species, count in species_count.items():
                st.write(f"• **{species}** — {count} times")

        elif summary_mode == "📉 Confusion Matrix":
            if pred_labels:
                labels = sorted(set(pred_labels))
                cm = confusion_matrix(true_labels, pred_labels, labels=labels)

                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                            xticklabels=labels, yticklabels=labels, cbar=False)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix")

                cols = st.columns([1, 2, 1])
                with cols[1]:
                    st.pyplot(fig)
            else:
                st.warning("No predictions to display confusion matrix.")
