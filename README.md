# Attention-Guided CNN–RNN Hybrid Model for Sequential Data Analysis

This repository contains the implementation of the research paper:

**“Attention-Guided CNN–RNN Hybrid Model for Sequential Data Analysis”**

The project explores a hybrid deep learning architecture that combines **Convolutional Neural Networks (CNNs)**, **Recurrent Neural Networks (RNNs)**, and an **Attention Mechanism** to improve feature selection and sequential modeling for sentiment analysis tasks.

---

## 📌 Overview

Deep learning architectures such as CNNs and RNNs are widely used for learning spatial and temporal patterns in sequential data. However, traditional CNN–RNN hybrid models propagate all extracted features into the recurrent layer without evaluating their importance.

This project introduces an **Attention-Guided Feature Refinement Module** that selectively emphasizes informative features before passing them to the RNN layer, improving classification performance and robustness.

Architecture:
Input → CNN → Attention Module → RNN (LSTM) → Fully Connected Layer → Output
