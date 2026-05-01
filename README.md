# Valuescope: Augmented Ensemble-Distillation Framework for Human Value Detection

## Overview
Valuescope is a real-time psychological health monitor and analytical framework for social media texts. Instead of relying on surface-level sentiment analysis (e.g., identifying if a post is "toxic" or "sad"), it utilizes a compressed, lightweight AI model to identify 19 deep human values driving a post, based on the Schwartz Theory of Basic Human Values.

By bridging deep psychological reasoning with geometric vector mathematics, Valuescope actively audits the psychological health of algorithmic feeds, measures user-feed friction, and tracks longitudinal shifts in mental health.

## Key Features & Analytical Engines
* **Unified 38-Dimensional Architecture:** A distilled student model (`RoBERTa-base`) that simultaneously predicts the presence of 19 core human values and their polarity (attained vs. constrained) in a single pass.
* **Value Dissonance Scoring Engine:** Mathematically calculates the psychological friction between a user's internal core values and the external content a recommendation algorithm pushes into their feed.
* **Longitudinal Value Drift Tracker:** Acts as a real-time computational biomarker, tracking the speed (magnitude) and geometric trajectory (direction) of a user's shifting ethical values over time.
* **Ethical Feed Scoring Engine:** Evaluates entire blocks of an algorithmic feed to generate an "Ethical Density Score," determining whether a digital space is psychologically balanced or an unhealthy echo chamber.

## Data Augmentation & The Psychological Translation Layer
Standard value datasets often suffer from extreme imbalance (e.g., an abundance of 'Achievement' data, but very little for 'Humility'). To solve this, we engineered a Psychological Translation Layer that maps raw emotional data from the GoEmotions dataset directly into psychological value-attainment pairs. This successfully balanced the dataset with over 94,000 diverse examples.

![Distribution of 19 Schwartz Values (Before vs After)](before_after_chart.png)

## Model Pipeline (Teacher-Student Distillation)
To overcome the extreme computational bottleneck of running massive ensemble models in real-time, Valuescope uses an **Ensemble Knowledge Distillation** process:
1. **The Teachers:** Two massive, high-performance models (`RoBERTa-Large` and `DeBERTa-Large`) are fine-tuned as domain experts.
2. **Logit Blending:** The system averages the predictions (logits) of both Teachers to create a smooth, highly reliable target signal.
3. **The Student:** A lightweight LLM (`RoBERTa-base`) is trained using this blended knowledge, learning complex psychological patterns it could never map on its own.

## Final Results & Real-Time Analytics
The distilled Student model successfully matched the F1 performance (~21.08%) of the massive Teacher models while remaining computationally light enough for continuous, real-time feed auditing.

The system translates the 38-dimensional tensor outputs back into 19 baseline probabilities to process through the geometric analytical engines. Below is a real-time snapshot of the engines tracking a simulated social media feed:

![Real-Time Analytics Dashboard Output](final_results.png)

*The Ethical Density Score (0.7873) confirms a relatively balanced psychological feed, while the Dissonance Score (0.0105) shows strong alignment between the user and the algorithm. The tracker also captured a continuous change vector with a magnitude of 0.1287 moving at a 149.0-degree trajectory.*

## Getting Started

### Prerequisites
* Python 3.8+
* PyTorch
* HuggingFace Transformers
* Scikit-Learn
* NumPy / Matplotlib

### Installation
```bash
git clone [https://github.com/yourusername/valuescope.git](https://github.com/yourusername/valuescope.git)
cd valuescope
pip install -r requirements.txt