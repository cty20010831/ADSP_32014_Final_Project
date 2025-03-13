# ADSP_32014_Final_Project - Bayesian Multimodal Fusion for Depression Classification   

## 📌 **Overview**
This project applies **Bayesian Machine Learning** techniques to classify depression using **speech embeddings from Wav2Vec 2.0**. The goal is to **quantify uncertainty** in predictions and enhance model interpretability for **clinical decision-making**.

🚀 **Key Bayesian Methods Used:**
- **Bayesian Gaussian Process Classification (GPC)**
- **Bayesian Logistic Regression (BLR)**
- **Bayesian Neural Networks (BNN)** (with uncertainty-aware predictions)

---

## 🎯 **Motivation**
Major Depressive Disorder (MDD) is often **underdiagnosed** due to subjective assessments.  
🔹 **Why Bayesian Learning?** It provides:
✅ **Uncertainty estimation** (crucial for clinical AI)  
✅ **Better generalization** over small datasets  
✅ **Robustness to noise** in speech data  

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

## 📊 **Methodology**
### **1️⃣ Data Processing**
The project focuses on audio data, from which we extract embeddings using Wav2Vec 2.0.
Preprocessing Steps:
- Resample audio to 16 kHz.
- Convert to mono.
- Normalize volume.
- Trim silence.
- Extract Wav2Vec 2.0 embeddings.

Dataset Details:
- Audio Experiment:
    - Participants: 52 participants (23 outpatients with depression, 29 healthy controls).
    - Recording Structure: Each subject provides 29 recordings:
	- Interview: Recordings 1–6 (positive), 7–12 (neutral), 13–18 (negative).
	- Short Story: Recording 19.
	- Reading: Recordings 20–21 (positive), 22–23 (neutral), 24–25 (negative).
	- Picture Description: Recordings 26–28.
	- TAT (Thematic Apperception Test): Recording 29.

Subgroup embeddings are computed for each recording group and later fused using an attention mechanism.

---

### **2️⃣ Bayesian Logistic Regression (BLR) - Sam**

This section describes our Bayesian model for depression classification using fused speech embeddings. The model is built in two main parts: **Attention Fusion** (to combine subgroup embeddings) and **Bayesian Logistic Regression** (to classify the fused embeddings). In addition, we perform posterior predictive checks on the training set and uncertainty quantification on the test set.

---

#### Bayesian Model Construction

##### Attention Fusion

- **Latent Attention Weights:**  
  For each training subject, we assign latent attention weights over the \( N \) subgroups using a Dirichlet prior. That is, for subject \( i \) (with \( i=1,\dots,M \)):
  \[
  \boldsymbol{\alpha}_i = (\alpha_{i1}, \dots, \alpha_{iN}) \sim \operatorname{Dirichlet}(\mathbf{1})
  \]
  where \(\mathbf{1}\) is an \(N\)-dimensional vector of ones.

- **Fused Embedding Computation:**  
  The fused embedding for each subject is computed as a weighted sum of the subgroup embeddings:
  \[
  \tilde{\mathbf{x}}_i = \sum_{j=1}^{N} \alpha_{ij}\,\mathbf{x}_{ij}
  \]
  where \(\mathbf{x}_{ij} \in \mathbb{R}^d\) is the embedding for subgroup \(j\) of subject \(i\).

##### Bayesian Logistic Regression

- **Model Specification:**
  - **Priors:**  
    \[
    \beta \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad b \sim \mathcal{N}(0, 1)
    \]
  - **Linear Predictor:**  
    \[
    z_i = \beta^\top \tilde{\mathbf{x}}_i + b
    \]
  - **Probability Transformation:**  
    \[
    \theta_i = \sigma(z_i) = \frac{1}{1+\exp(-z_i)}
    \]
  - **Likelihood:**  
    \[
    y_i \sim \operatorname{Bernoulli}(\theta_i)
    \]
  
The model jointly infers the attention weights \(\boldsymbol{\alpha}_i\) and the logistic regression parameters (\(\beta\) and \(b\)), providing both point predictions and uncertainty estimates for each subject.

---

#### Posterior Predictive Checks on Training Data

- **Purpose:**  
  To visually assess whether the model's predicted probabilities resemble the distribution of the observed data.

- **Process:**  
  - Generate posterior predictive samples (i.e., simulated values of \(\theta_i\)) for the training set.
  - Plot histograms and density plots of the predicted probabilities.
  
*Example Visualizations:*
- **Histogram of Predicted Probabilities:** Helps verify that the distribution of predictions covers the range of possible values.
- **Density Plot by Class:** Overlays the predicted probability distributions for subjects with \(y_i = 1\) (MDD) and \(y_i = 0\) (healthy controls) to evaluate class separation.

---

#### Uncertainty Quantification and Model Evaluation on Test Data

##### Fused Embeddings on Test Data

- **Approach:**  
  Since the model is trained on the training set, we use the global mean of the training attention weights to compute fused embeddings for test subjects. This gives a consistent representation of the test data:
  \[
  \tilde{\mathbf{x}}^{\text{test}}_i = \sum_{j=1}^{N} \bar{\alpha}_{j}\,\mathbf{x}^{\text{test}}_{ij}
  \]
  where \(\bar{\alpha}_{j}\) is the average attention weight over the training subjects for subgroup \(j\).

##### Prediction and Uncertainty

- **Posterior Sampling:**  
  We sample posterior values for \(\beta\) and \(b\) from our training model.
  
- **Prediction:**  
  For each test subject, we compute the fused embedding and then the predicted probability:
  \[
  z_i^{\text{test}} = \beta^\top \tilde{\mathbf{x}}^{\text{test}}_i + b, \quad \theta_i^{\text{test}} = \sigma(z_i^{\text{test}})
  \]
  
- **Uncertainty Quantification:**  
  By drawing multiple posterior samples and computing predictions, we obtain a distribution of predicted probabilities for each test subject. The standard deviation of this distribution reflects the uncertainty in the prediction.

##### Model Evaluation

- **Metrics on Test Data:**  
  We evaluate the model using:
  - **Accuracy**
  - **AUC-ROC**
  - **Precision, Recall, and F1-Score**
  - **Brier Score**
  
- **Calibration:**  
  A calibration curve is plotted to assess if the predicted probabilities are well-calibrated (i.e., when the model predicts 70% probability, the true frequency of MDD is close to 70%).

---

### **3️⃣ Bayesian Gaussian Process Classifier (GPC) - Irene**
Gaussian Process Classification is a **non-parametric Bayesian approach** used for depression classification.  
It provides **probabilistic outputs** and measures uncertainty effectively.

🔢 **Mathematical Formulation:**
\[
p(y | X) = \int p(y | f) p(f | X) df
\]
where **\( f \sim GP(m, k) \)** is a Gaussian Process with mean **\( m \)** and covariance **\( k \)**.

📌 **Results:**  
✅ **Accuracy: 94.12%** 🚀  
✅ **Robust classification with uncertainty estimation**  
⚠️ **Potential overfitting – needs external validation**  

---

### **4️⃣ Bayesian Neural Networks (BNN) with Threshold Tuning - Irene**
Bayesian Neural Networks (BNN) were trained, and we optimized the **decision threshold** to balance **precision and recall**.

#### **Threshold Experimentation Results:**
| **Threshold** | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|--------------|-------------|--------------|-------------|--------------|
| **0.22**  | **40.00%** | **42.11%** | **88.89%** | **57.14%** |
| **0.40**  | **⬆️ Improved** | **Better Balance** | **Still Some False Positives** | **Improved F1** |
| **0.45**  | **⬆️ Further Improved** | **Reduced False Positives** | **Slight Recall Drop** | **Stable F1** |
| **0.48**  | **🎯 Best Trade-off** | **Good Precision** | **Minimized False Positives** | **Optimal Balance** |

✅ **Final Decision:** **Threshold = 0.48 provided the best balance.**  
🖼 **Precision-Recall Curve:**  
![Precision-Recall Curve](path/to/your_image2.png)

---