# Classification Metrics: Definitions & Intuition

When building classifiers like the FairFace skin tone model, accuracy alone doesn't tell the full story—especially when classes are unbalanced. We use Precision, Recall, and the F1-Score to understand *how* the model fails.

---

## 1. The Core Concepts: True/False Positives & Negatives

Imagine we're evaluating how well the model predicts **"Very Dark" (MST 9-10)** skin:

* **True Positive (TP):** The image *is* Very Dark, and the model *predicted* Very Dark. (Correct)
* **True Negative (TN):** The image *is not* Very Dark, and the model *predicted* something else. (Correct)
* **False Positive (FP):** The model *predicted* Very Dark, but the image is actually Light/Medium. (Type I Error / "False Alarm")
* **False Negative (FN):** The image *is* Very Dark, but the model *predicted* Medium/Light. (Type II Error / "Miss")

---

## 2. Precision: "When you predicted X, how often were you right?"

**Mathematical Definition:**

$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

**Intuition:** Precision tells you the *purity* of your predictions. If the model says an image is "Very Dark," how much can you trust it?

* **High Precision:** If it predicts Very Dark, it's almost certainly Very Dark (few false alarms).
* **Low Precision:** It's "trigger-happy" and predicting Very Dark for images that are actually Medium or Light.

*Example from v3.0:* The "Very Light" class has a low precision of **0.59**. This means when the model outputs "Very Light", it's wrong 41% of the time (often mislabeling Light or Very Dark images as Very Light).

---

## 3. Recall: "Out of all actual X, how many did you find?"

**Mathematical Definition:**

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

**Intuition:** Recall tells you the *completeness* of your predictions. Does the model identify every genuine instance of the class?

* **High Recall:** It correctly identifies almost all the Very Dark images in the dataset (few misses).
* **Low Recall:** It is overly cautious and frequently mislabels Very Dark images as something else.

*Example from v3.0:* The "Very Dark" class has a high recall of **0.90**. This means out of all the true Very Dark images in the dataset, the model successfully found 90% of them.

> **The Trade-off:** Precision and Recall often inverse each other. If you make a model very strict (only say "Very Dark" if it's absolutely sure), Precision goes up but Recall goes down. If you make it very loose, Recall goes up but Precision goes down.

---

## 4. F1-Score: The Balancing Act

**Mathematical Definition:** The harmonic mean of Precision and Recall.

$$
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Intuition:** You only get a high F1-score if **both** Precision and Recall are high. It's the best single number to look at to judge a model's performance on a specific class, preventing you from being fooled by a model that just guesses the majority class every time.

---

## 5. Macro vs. Weighted Averages

When evaluating all 5 classes together, how do we combine their F1-scores?

* **Macro Average:** Calculate the F1-score for each of the 5 classes independently, then take the unweighted average.
  * *Why it matters:* It treats "Very Light" (341 samples) equally to "Dark" (4,793 samples). **This is the most important metric for fairness,** as it heavily penalises the model if it performs poorly on minority groups.
* **Weighted Average:** Calculate the F1-score for each class, but weight their contribution by the number of samples in the dataset.
  * *Why it matters:* It tells you the expected performance on a random sample from your dataset, but it can hide terrible performance on minority classes.
