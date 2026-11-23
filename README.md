# SignVerify: AI-Powered Offline Signature Verification

A deep learning college project for offline signature verification using a Siamese Convolutional Neural Network (CNN) with PyTorch and Streamlit.

## Features
- **Siamese CNN Architecture**: Learns feature representations to compute similarity distances between genuine and test signatures.
- **Contrastive Loss**: Trained using pair-wise contrastive loss to distinguish genuine signatures from skilled forgeries.
- **Interactive Streamlit Web App**: Clean, intuitive user interface for uploading reference signatures and testing test signatures in real-time.
- **Automated Verification**: Computes pairwise distances against a calibrated threshold to classify signatures as `GENUINE` or `FORGED`.

## Project Structure
- `app.py`: Streamlit application for interactive signature verification.
- `siamese_model.py`: Siamese Convolutional Neural Network architecture definition.
- `contrastive_loss.py`: Contrastive loss function implementation.
- `train.py`: Training script for the Siamese network.
- `preprocess.py`: Image preprocessing utilities.
- `siamese_dataset.py`: PyTorch Dataset class for signature pairs.
- `evaluate.py` / `evaluate_clean_test.py`: Evaluation and testing scripts.
- `find_threshold.py` / `find_threshold_v3.py`: Scripts to determine optimal classification threshold.
- `data/`: Dataset splits and metadata CSV files.
- `siamese_model_v3.pth`: Pretrained Siamese CNN model weights.

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Streamlit Web App**:
   ```bash
   streamlit run app.py
   ```
