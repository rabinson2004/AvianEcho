
# 🕊️ Avian Echo – Bird Recognition and Counting App

**Avian Echo** is a Streamlit-based deep learning application that identifies and counts bird species from audio recordings using a trained EfficientNet-B2 model and log-mel spectrogram analysis.

---

## 🚀 Features

- 🔍 **Bird Sound Classification** using a fine-tuned EfficientNet-B2 model.
- 🎧 **Supports MP3/WAV audio files** for real-time bird species prediction.
- 📈 **Two Modes**:  
  - **Single Bird**: Extracts and classifies the best-sounding 10-second segment.  
  - **Multiple Birds**: Segments audio with overlap, performs per-segment predictions, and provides visual summaries.
- 🎼 **Log-Mel Spectrogram** visualization for each prediction segment.
- 🖼️ **Bird Image Mapping**: Displays corresponding bird images if available.
- 🧠 **Confidence Score Display** for each prediction.
- 📊 **Detection Summary Dashboard**:
  - Time-wise bird prediction log
  - Most frequently detected bird
  - Unique bird species detected
  - Confusion matrix (for visualizing prediction patterns)

---

## 🧠 Model Details

- Architecture: **EfficientNet-B2**
- Input: **Mel Spectrogram Images**
- Trained with: **16 Indian bird species**
- Format: PyTorch checkpoint `.pth` with:
  - `model_state_dict`
  - `class_to_idx`
  - `transform`

---

## 📂 Project Structure

```
AvianEcho/
│
├── bird_images/                  # Directory with bird images (jpg/png)
├── best_bird_model.pth          # Trained PyTorch model checkpoint
├── app.py                       # Main Streamlit app
├── temp_audio.*                 # Temporarily saved uploaded audio
├── temp_spectrogram.png         # Temp spectrogram for prediction
├── best_spectrogram.png         # Spectrogram of best segment (Single Bird)
├── segment_spectrogram.png      # Segment spectrogram (Multiple Birds)
```

---

## 🛠️ Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/avian-echo.git
   cd avian-echo
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

---

## 🔧 Dependencies

- `streamlit`
- `torch`
- `librosa`
- `numpy`
- `matplotlib`
- `Pillow`
- `scikit-learn`
- `seaborn`

You can create a `requirements.txt` with:

```txt
streamlit
torch
librosa
numpy
matplotlib
Pillow
scikit-learn
seaborn
```

---

## 📸 Screenshots

### 🎧 Upload Audio
![Upload](screenshots/upload_audio.png)

### 🔍 Single Bird Prediction
![Single](screenshots/single_bird.png)

### 🦜 Multiple Bird Segments
![Multi](screenshots/multi_bird_segments.png)

### 📊 Summary and Confusion Matrix
![Confusion](screenshots/confusion_matrix.png)

---

## 📁 Dataset

- 16 bird species
- Each species has 600 training, 75 validation, and 75 test mel spectrogram images.
- Audio source: [Xeno-Canto](https://xeno-canto.org)

---

## 🤖 Future Enhancements

- Add real-time microphone recording.
- Use DBSCAN for improved multi-bird event grouping.
- Improve UI with animations and responsive layout.
- Add species information using Wikipedia API.

---

## 👨‍🔬 Authors

- A. Hamilton Infant  
- G. Rabinson  
- M.J. Theo Savio  
**Guide**: Ms. T. Anitha Dorothy  
**Affiliation**: Saranathan College of Engineering, Trichy – 620012  
**Department**: Artificial Intelligence and Data Science

---

## 📜 License

This project is for academic and educational purposes. Please cite appropriately if reused.
