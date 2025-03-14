# Sam

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

This section describes our Bayesian model for depression classification using fused speech embeddings. The model is built in two main parts: **Attention Fusion** (to combine subgroup embeddings) and **Bayesian Logistic Regression** (to classify the fused embeddings). In addition, we perform posterior predictive checks on the training data and uncertainty quantification on the test data.

---

#### Bayesian Model Construction

##### Attention Fusion

- **Latent Attention Weights:**  
  For each training subject $i$ (with $i=1,\dots,M$)), we assign latent attention weights over the $N$ subgroups using a Dirichlet prior:
  $
  \boldsymbol{\alpha}_i = (\alpha_{i1}, \dots, \alpha_{iN}) \sim \operatorname{Dirichlet}(\mathbf{1})
  $, 
  where $\mathbf{1}$ is an $N$-dimensional vector of ones.

- **Fused Embedding Computation:**  
  The fused embedding for each subject is computed as a weighted sum of the subgroup embeddings:
  ```math
  \tilde{\mathbf{x}}_i = \sum_{j=1}^{N} \alpha_{ij}\,\mathbf{x}_{ij}
  ```
  Here, $\mathbf{x}_{ij} \in \mathbb{R}^d$ is the embedding for subgroup $j$ of subject $i$.

#### Bayesian Logistic Regression

- **Model Specification:**
  - **Priors:**  
    ```math
    \beta \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \quad \text{and} \quad b \sim \mathcal{N}(0, 1)
    ```
  - **Linear Predictor:**  
    ```math
    z_i = \beta^\top \tilde{\mathbf{x}}_i + b
    ```
  - **Probability Transformation:**  
    ```math
    \theta_i = \sigma(z_i) = \frac{1}{1+\exp(-z_i)}
    ```
  - **Likelihood:**  
    ```math
    y_i \sim \operatorname{Bernoulli}(\theta_i)
    ```

The model jointly infers the latent attention weights $\boldsymbol{\alpha}_i$ and the logistic regression parameters ($\beta$ and $b$), providing both point predictions and uncertainty estimates for each subject.

---

#### Posterior Predictive Checks on Training Data

- **Objective:**  
  To visually assess whether the model’s predicted probabilities resemble the distribution of the observed data.

- **Steps:**
  - Generate posterior predictive samples for the training set (simulated values of $\theta_i$).
  - Compute the average predicted probability for each subject.
  - Visualize the distribution using:
    - **Histogram:** To see the overall distribution of predicted probabilities.
    - **Density Plot by Class:** To compare the distribution of predictions for subjects with $y_i = 1$ (MDD) and $y_i = 0$ (healthy controls).

---

#### Uncertainty Quantification and Model Evaluation on Test Data

##### Fused Embeddings on Test Data

- **Approach:**  
  Since the model is trained on the training data, we use the global mean of the training attention weights to compute fused embeddings for test subjects. This is given by:
  ```math
  \tilde{\mathbf{x}}^{\text{test}}_i = \sum_{j=1}^{N} \bar{\alpha}_{j}\,\mathbf{x}^{\text{test}}_{ij}
  ```
  where $\bar{\alpha}_{j}$ is the average attention weight over the training subjects for subgroup $j$.

##### Prediction and Uncertainty

- **Posterior Sampling:**  
  We sample posterior values for \(\beta\) and \(b\) from the training model.

- **Prediction:**  
  For each test subject, the fused embedding is computed and the logistic regression model produces a predicted probability:
  ```math
  z_i^{\text{test}} = \beta^\top \tilde{\mathbf{x}}^{\text{test}}_i + b, \quad \theta_i^{\text{test}} = \sigma(z_i^{\text{test}})
  ```
  
- **Uncertainty Quantification:**  
  By drawing multiple posterior samples and computing the predicted probability for each test subject, we obtain a distribution for each prediction. The standard deviation of this distribution indicates the uncertainty in the model's prediction.

##### Model Evaluation

- **Calibration:**  
  A calibration curve is plotted to assess whether the predicted probabilities are well-calibrated. For instance, if the model predicts a probability of 0.70, the observed frequency of MDD should be close to 70%.

- **Metrics on Test Data:**  
  We evaluate the model's performance using:
  - Accuracy
  - AUC-ROC
  - Precision, Recall, and F1-Score
  - Brier Score

---


## Virtual Environment

To replicate this analysis, you can use the provided `requirements.txt` in a Python 3.11 environment:

```bash
# Create the virtual environment:
python3.13 -m venv venv

# Activate the virtual environment:
source venv/bin/activate

# Install required packages:
python3 -m pip install -r requirements.txt
```

## Running the Analysis
Simply run `analysis.ipynb` to reproduce the analysis.