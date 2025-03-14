# ADSP_32014_Final_Project - Bayesian Multimodal Fusion for Depression Classification   

## 📌 **Overview**
This project applies **Bayesian Machine Learning** techniques to classify depression using **speech embeddings from Wav2Vec 2.0** and **features extracted from EEG signals**. The goal is to **quantify uncertainty** in predictions and enhance model interpretability for **clinical decision-making**.

🚀 **Key Bayesian Methods Used:**
- **Bayesian Logistic Regression (BLR)**
- **Bayesian Gaussian Process Classification (GPC)**
- **Bayesian Neural Networks (BNN)** (with uncertainty-aware predictions)

---

## 🎯 **Motivation**
Major Depressive Disorder (MDD) is often **underdiagnosed** due to subjective assessments.  

🔹 **Why Bayesian Learning?** It provides:
 - ✅ **Uncertainty estimation** (crucial for clinical AI)  
 - ✅ **Better generalization** over small datasets  
 - ✅ **Robustness to noise** in speech data  

---

## 📂 **Dataset**

### Overview

We utilize an open‐source multimodal dataset for mental‐disorder analysis as described in [Cai et al., 2022](https://doi.org/10.1038/s41597-022-01211-x). This dataset contains EEG recordings and spoken language data collected from clinically depressed patients (MDD) and matching healthy controls (HC). The dataset was carefully curated by clinical experts and includes both physiological signals and behavioral assessments, providing a comprehensive foundation for developing machine learning models for depression classification.

### Data Components

The dataset is organized into three main components:

#### 1. Full Brain 128-Electrodes EEG Data
- **Acquisition:**  
  - Collected using a traditional 128-electrodes mounted elastic cap (HydroCel Geodesic Sensor Net, Electrical Geodesics Inc., Oregon, USA).
- **Tasks:**  
  - **Resting State:** Five minutes of eyes-closed resting-state EEG.
  - **Dot-Probe Task:** EEG recordings during a dot-probe experiment that assesses attentional bias with facial stimuli.
- **Participants:**  
  - 53 subjects (24 diagnosed with MDD, 29 healthy controls).

#### 2. 3-Electrodes EEG Data
- **Acquisition:**  
  - Recorded using a wearable 3-electrode EEG collector designed for pervasive computing.
- **Task:**  
  - Resting-state EEG.
- **Participants:**  
  - 55 subjects (26 diagnosed with MDD, 29 healthy controls).

#### 3. Recordings of Spoken Language
- **Acquisition:**  
  - Audio recordings in uncompressed WAV format, captured in a quiet, soundproof environment.
- **Tasks:**  
  - **Interview:** 18 questions covering positive, neutral, and negative emotional content (e.g., travel plans, gifts, self-evaluation, sleep difficulties).
  - **Reading:** A short story ("The North Wind and the Sun") and readings of word groups with distinct emotional valences (positive, neutral, negative).
  - **Picture Description:** Participants describe images (four images in total), with three images selected for positive, neutral, and negative expressions and one image from the Thematic Apperception Test (TAT).
- **Participants:**  
  - 52 subjects (23 diagnosed with MDD, 29 healthy controls).

### Experimental Protocol

- **Participant Selection:**  
  Participants were carefully diagnosed by professional psychiatrists using structured interviews such as the Mini-International Neuropsychiatric Interview (MINI) and clinical scales like the PHQ-9. Inclusion and exclusion criteria ensured high-quality data collection.
  
- **Data Recording:**  
  - **EEG Data:** Recorded at a sampling rate of 250 Hz. For full-brain EEG, impedance was maintained below 50 kΩ. Data are provided in multiple formats (e.g., .mff, .mat, .EDF) and in BIDS format.
  - **Audio Data:** Recorded using high-fidelity microphones (e.g., Neumann TLM102) at 44.1 kHz and 24-bit depth. Recordings were manually segmented and labeled.
  
- **Data Storage:**  
  The dataset is available as safeguarded data on the UK Data Archive’s ReShare repository. Access requires registration with the UK Data Service and agreement to their End User License conditions.

### Accessing the Dataset

The dataset can be downloaded from the publicly accessible repository after registration:
- [UK Data Service ReShare](https://ukdataservice.ac.uk/cd137-enduserlicence/)

It is also available at:
- [MODMA Dataset](http://modma.lzu.edu.cn/data/index/)

### Reference

For full details, please refer to the original publication:

**Cai, H., Yuan, Z., Gao, Y., Sun, S., Li, N., Tian, F., et al. (2022). A multi-modal open dataset for mental-disorder analysis. _Scientific Data, 9_:178.**  
DOI: [10.1038/s41597-022-01211-x](https://doi.org/10.1038/s41597-022-01211-x)

---
