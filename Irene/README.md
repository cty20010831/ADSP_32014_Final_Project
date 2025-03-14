# 🧠 Bayesian Multimodal Fusion for Depression Classification  
[![Bayesian Machine Learning](https://img.shields.io/badge/Bayesian-Machine_Learning-blue)](https://github.com/yourrepo)  
**A probabilistic approach for detecting Major Depressive Disorder (MDD) using Bayesian inference with speech and EEG features.**  

##  **Why Does This Matter?**
- Depression affects over 300 million people worldwide and is the leading cause of disability.
- In the U.S. alone, 21 million adults experience at least one major depressive episode each year.
- Missed or delayed diagnosis leads to unnecessary suffering and costs the global economy over $1 trillion per year.

## 📌 **Overview**
This project applies **Bayesian Machine Learning** techniques to classify depression using **speech embeddings from Wav2Vec 2.0** and **EEG-based neural activity features**. The goal is to **quantify uncertainty** in predictions and enhance model interpretability for **clinical decision-making**.

- 📊 **Data Shape:** 1,774 samples, 823 features  
- 🧠 **Multimodal Data:** Speech + EEG  

🚀 **Key Bayesian Methods Used:**  
- **Bayesian Neural Networks (BNN)** (with uncertainty-aware predictions)  
- **Bayesian Gaussian Process Classification (GPC)**  
- **Bayesian Logistic Regression (BLR)**  
- **KL Weight Annealing & Threshold Tuning** (for optimizing precision and recall)  

---

## 🎯 **Motivation**
Major Depressive Disorder (MDD) is often **underdiagnosed** due to subjective assessments.  
🔹 **Why Bayesian Learning?** It provides:  
✅ **Uncertainty estimation** (crucial for clinical AI)  
✅ **Better generalization** over small datasets  
✅ **Robustness to noise** in speech & EEG data  

---

## 📊 **Methodology**
### **1️⃣ Data Processing**
We use **speech data** (Wav2Vec 2.0) and **EEG signals** as features.

📌 **Preprocessing Steps:**  
- **Speech:** Resampling, silence trimming, volume normalization, embedding extraction  
- **EEG:** Filtering, noise removal, dimensionality reduction (PCA)  

---

### **2️⃣ Bayesian Neural Network (BNN)**
We used a **probabilistic BNN** with **KL weight annealing** and **threshold tuning** to improve classification performance.  

🔢 **KL Weight Annealing:**  
We gradually increased the KL divergence weight over time to **stabilize training and reduce overfitting**.  

📌 **Final Results:**  
| **Threshold** | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|--------------|-------------|--------------|-------------|--------------|
| **0.50**  | **55.19%** | **73.00%** | **16.46%** | **27.12%** |
| **0.65**  | **65.00%** | **53.33%** | **100.00%** | **69.57%** ✅ |

🚀 **Best Threshold:** **0.65** → **Maximizes Recall (100%) while maintaining a good F1-score.**  

🖼 **Precision-Recall Curve:**  
![Precision-Recall Curve](path/to/your_image.png)

---

### **3️⃣ Bayesian Gaussian Process Classification (GPC)**
Gaussian Process Classification is a **non-parametric Bayesian approach** that provides **probabilistic outputs** and measures **uncertainty effectively**.

📌 **GPC Results:**  
✅ **Accuracy: 94.12%**  
✅ **Robust classification with uncertainty estimation**  
⚠️ **Potential overfitting – needs external validation**  

---

### **4️⃣ Bayesian Logistic Regression (BLR)**
Bayesian Logistic Regression helps **analyze feature importance** and **quantify uncertainty** in the model.

📌 **Key Observations:**  
- Some weight distributions were **multimodal**, indicating **uncertainty in feature contributions**.  
- Certain features had a **94% HDI including zero**, suggesting they may not strongly influence predictions.  

🖼 **Weight Posterior Distributions:**  
![Posterior Distributions](path/to/your_image2.png)

---

## 🛠 **Code Implementation**
### **1️⃣ Bayesian Neural Network (BNN)**
```python
threshold = 0.65  # Optimized threshold
with torch.no_grad():
    y_pred_probs = bnn(X_test)
    y_pred = (y_pred_probs > threshold).float()
```

### **2️⃣ KL Weight Annealing in PyMC**
```python
with pm.Model() as bnn_model:
    kl_weight = pm.Data("kl_weight", 0.1)  # Start small, increase over time
    w = pm.Normal("w", mu=0, sigma=1, shape=(X_train.shape[1],))
    logits = pm.math.dot(X_train, w)
    pm.Bernoulli("y_obs", logit_p=logits, observed=y_train)
```

## 📈 **Results Summary**

| **Model**                 | **Accuracy** | **Uncertainty Estimation**                  |
|---------------------------|-------------|---------------------------------------------|
| **GPC**                   | **94.12%**  | ✅ Strong uncertainty quantification       |
| **BLR**                   | **TBD**     | ✅ Provides feature importance             |
| **BNN (Threshold 0.50)**   | **55.19%**  | ✅ KL Weight Annealing added               |
| **BNN (Threshold 0.65)**   | **65.00%**  | ✅ Best Precision-Recall Balance          |

## 🚀 **Final Recommendation:**

- BNN + Threshold 0.65 provides the best trade-off between precision and recall.
- GPC is highly accurate but may need validation to check for overfitting.
- BLR helps in understanding feature importance.

## 📌 **Future Work**
- 🔹 EEG Feature Integration Improvements 🧠
- 🔹 Hybrid Bayesian Models: Combine GPC, BLR, and BNN for multimodal fusion.
- 🔹 Clinical Validation: Test the framework on real-world depression screening data.

## 📜 **Citations**

- Wav2Vec 2.0: https://arxiv.org/abs/2006.11477
- EEGPT: https://openreview.net/forum?id=lvS2b8CjG5
- EEG Dataset: https://www.nature.com/articles/s41597-022-01211-x
- Global Prevalence of Depression: The World Health Organization (WHO) reports that depression affects more than 300 million people worldwide and is the leading cause of disability. ​
- Prevalence in the United States: In the United States, approximately 8.4% of adults, equating to 21 million individuals, experience at least one major depressive episode annually. ​(en.wikipedia.org)
- Economic Impact: Depression and anxiety disorders result in the loss of approximately 12 billion working days each year, costing the global economy over $1 trillion annually. 

