# 🧠 Bayesian Multimodal Fusion for Depression Classification  
[![Bayesian Machine Learning](https://img.shields.io/badge/Bayesian-Machine_Learning-blue)](https://github.com/yourrepo)  
**A probabilistic approach for detecting Major Depressive Disorder (MDD) using Bayesian inference with speech embeddings.**  

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

## 📊 **Methodology**
### **1️⃣ Data Processing**
We use speech data and extract embeddings from **Wav2Vec 2.0**.

**Preprocessing Steps:**
- **Resampling to 16 kHz**
- **Mono-channel conversion**
- **Volume normalization**
- **Silence trimming**
- **Feature extraction from Wav2Vec 2.0 embeddings**

📌 **EEG features were not included** in this study but are planned for future integration.

---

### **2️⃣ Bayesian Gaussian Process Classifier (GPC)**
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

### **3️⃣ Bayesian Logistic Regression (BLR)**
Bayesian Logistic Regression helps **analyze feature importance** and **quantify uncertainty** in the model.

📌 **Key Observations:**
- **Some weight distributions were multimodal**, indicating **uncertainty in feature contributions**.
- **Certain features had a 94% HDI including zero**, suggesting they may not strongly influence predictions.

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
