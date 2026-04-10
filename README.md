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

## RESULT GRAPHS

<img width="567" height="444" alt="image" src="https://github.com/user-attachments/assets/facbd2ce-e4c9-4b33-a981-df22b9c52a58" />
<img width="557" height="447" alt="image" src="https://github.com/user-attachments/assets/e1be1f1f-0158-43e3-b984-2461c1cc08d9" />
<img width="553" height="433" alt="image" src="https://github.com/user-attachments/assets/0cdec73b-7d6f-49df-908d-678bff864460" />

## RESULT from code formatted in excel

<img width="312" height="225" alt="image" src="https://github.com/user-attachments/assets/eac16d89-4bc7-414a-8f41-6da46f012bec" />
<img width="322" height="111" alt="image" src="https://github.com/user-attachments/assets/8c788ca2-2a58-437b-bae2-92078dae0df0" />


