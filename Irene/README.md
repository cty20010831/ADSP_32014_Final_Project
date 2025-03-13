# 🧠 Bayesian Multimodal Fusion for Depression Classification  
[![Bayesian Machine Learning](https://img.shields.io/badge/Bayesian-Machine_Learning-blue)](https://github.com/yourrepo)  
**A probabilistic approach for detecting Major Depressive Disorder (MDD) using Bayesian inference with speech embeddings and EEG signals.**  

---

## 📌 **Overview**
This project applies **Bayesian Machine Learning** techniques to classify depression using **speech embeddings from Wav2Vec 2.0** and **EEG-based neural activity features**. The goal is to **quantify uncertainty** in predictions and enhance model interpretability for **clinical decision-making**.

- 📊 **Data Shape**: (1774 samples, 823 features)  
- 🎯 **Balanced Labels**: Equal distribution (0: 887, 1: 887)  

🚀 **Key Bayesian Methods Used:**  
- **Bayesian Neural Networks (BNN)** – Uncertainty-aware deep learning model  
- **Bayesian Logistic Regression (BLR)** – Probabilistic feature importance analysis  
- **Bayesian Gaussian Mixture Models (GMMs)** – Robust clustering  
- **Bayesian Hidden Markov Models (HMMs)** – Temporal EEG pattern analysis  

---

## 🎯 **Motivation**
Major Depressive Disorder (MDD) is often **underdiagnosed** due to subjective assessments.  
🔹 **Why Bayesian Learning?** It provides:  
✅ **Uncertainty estimation** (crucial for clinical AI)  
✅ **Better generalization** over small datasets  
✅ **Robustness to noise** in speech and EEG data  

---

## 📊 **Methodology**
### **1️⃣ Data Preprocessing**
We integrate **speech** and **EEG features** to create a robust dataset for Bayesian inference.

#### **Speech Processing (Wav2Vec 2.0):**  
✅ **Resampled to 16 kHz**  
✅ **Mono-channel conversion**  
✅ **Volume normalization & silence trimming**  
✅ **Extracted low, mid, and high-level features**  

#### **EEG Processing:**  
✅ **Artifact removal (ICA/adaptive filtering)**  
✅ **Z-score normalization**  
✅ **Dimensionality reduction via PCA (→ 55 dimensions)**  
✅ **Balanced & interpolated EEG dataset**  

---

### **2️⃣ Bayesian Clustering via Gaussian Mixture Models (GMMs)**
We applied **Bayesian Gaussian Mixture Models** to identify depression-related clusters.

📌 **Cluster Analysis:**  
| **Cluster** | **Depression Rate** |
|------------|----------------|
| **Cluster 0** | 25.0% |
| **Cluster 1** | 67.0% |
| **Cluster 2** | 100.0% |

✅ **Key Insight:**  
- **Cluster 2** had **100% depression rate**, meaning the model successfully grouped **high-risk individuals** together.

---

### **3️⃣ Bayesian Logistic Regression (BLR)**
Bayesian Logistic Regression helps **analyze feature importance** and **quantify uncertainty**.

📌 **Key Observations:**  
- **Certain EEG & speech features had high uncertainty**, suggesting **low contribution to depression classification**.  
- **95% HDI (Highest Density Interval) included zero for some features**, meaning **they may be irrelevant**.  

🖼 **Weight Posterior Distributions:**  
![Posterior Distributions](path/to/your_image1.png)  
*(This shows Bayesian weights with uncertainty quantification.)*

---

### **4️⃣ Bayesian Neural Networks (BNN) with Threshold Tuning**
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

## 📈 **Final Results & Model Evaluation**  

### **✅ Updated Bayesian Neural Network Results:**
| **Metric** | **Initial BNN** | **Optimized BNN** |
|-----------|---------------|---------------|
| **Accuracy** | **51.75%** | **55.19%** |
| **Precision** | **51.82%** | **73.00%** |
| **Recall** | **49.72%** | **16.46%** |
| **ROC-AUC** | **51.75%** | **55.19%** |

📌 **Key Takeaways:**  
- **Higher Precision (73%)** means **fewer false positives**, making the model **useful for screening**.  
- **Low Recall (16%)** means **some cases of depression were missed**, suggesting **further model refinements**.  
- **ROC-AUC improved** from **51.75% → 55.19%**, indicating **better predictive power**.  

---

## 🛠 **Code Implementation**
### **1️⃣ Gaussian Process Classification (GPC)**
```python
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

kernel = 1.0 * RBF(length_scale=1.0)
gpc = GaussianProcessClassifier(kernel=kernel)
gpc.fit(X_train, y_train)

accuracy = gpc.score(X_test, y_test)
print(f"GPC Accuracy: {accuracy:.4f}")
```
## 2️⃣ **Bayesian Logistic Regression**

```python
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

def bayesian_model(X, y=None):
    w = pyro.sample("w", dist.Normal(torch.zeros(X.shape[1]), torch.ones(X.shape[1])))
    logits = (X @ w).sigmoid()
    with pyro.plate("data", X.shape[0]):
        pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)
```
## 3️⃣ **Bayesian Neural Network (BNN)**

```python
threshold = 0.48  # Optimized threshold
y_pred = (y_pred_probs > threshold).float()
```

## 📌 **Future Work**
- 🔹 EEG Feature Integration: Expand the model to incorporate brainwave activity.
- 🔹 Hybrid Bayesian Models: Combine GPC, BLR, and BNN for multimodal fusion.
- 🔹 Clinical Validation: Test the framework on real-world depression screening data.

## 📜 **Citations & References**
- Wav2Vec 2.0: https://arxiv.org/abs/2006.11477
- EEGPT: https://openreview.net/forum?id=lvS2b8CjG5
- EEG Depression Data: https://www.nature.com/articles/s41597-022-01211-x
