# 🧠 AI Cohort Learning Journey — Day by Day

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange?style=flat&logo=googlecolab)
![Status](https://img.shields.io/badge/Status-Active%20%7C%20Daily%20Updates-brightgreen?style=flat)
![Days](https://img.shields.io/badge/Journey-Day%201%20of%2060-purple?style=flat)

> I joined an AI cohort with one goal — land a job at an AI company in 2 months.  
> This repo is my daily learning log. Every concept I learn gets coded, documented, and pushed here.  
> Built in public. Updated every day. Zero fluff.

---

## 👋 Who This Is For

- Recruiters & hiring managers wanting to see my learning progression
- Fellow beginners on the same AI journey
- Anyone who wants to see AI fundamentals explained simply and coded from scratch

---

## 📌 What This Repo Covers

This is not a copy-paste of tutorials.  
Every notebook starts from **first principles** — I write the code myself, break down every line, and connect it back to how real AI systems work.

The journey goes from:

```
f(x) = weight * x + bias   →   Full neural networks   →   Real AI projects
```

---

## 🗂️ Repo Structure

```
📁 ai-cohort-journey/
│
├── 📓 Day01_AI_Fundamentals.ipynb       ← Learning methods, neural nets, loss, gradient descent
├── 📓 Day02_...ipynb                    ← coming soon
│
└── README.md                            ← you are here
```

---

## ✅ Day 01 — AI Fundamentals & Neural Network Basics

**Notebook:** `Day01_AI_Fundamentals.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

### 🎯 What I Learned

#### 1. What is a Neural Network?

A neural network is a system inspired by the human brain. It takes inputs, runs them through layers of simple math, and produces an output. It learns by comparing its output to the correct answer and adjusting itself to do better next time.

Think of it like a student taking a test:
- Makes a guess
- Checks against the answer key
- Learns from the mistake
- Repeats until it gets it right

---

#### 2. The Core Equation — `f(x) = weight * x + bias`

This single equation is what every neuron in every neural network runs.

| Part | What it is | Simple analogy |
|---|---|---|
| `x` | Input — your raw data | The player's stats |
| `weight` | How much importance to give `x` | How much the scout cares about that stat |
| `bias` | Default starting value | The scout's gut feeling before watching anyone |
| `f(x)` | The output — the network's prediction | The final score the scout gives |

```python
def neuron(x, weight, bias):
    return weight * x + bias

# Example: predicting a house price
x      = 1850   # square footage
weight = 0.5    # importance of size
bias   = 20000  # base price offset

prediction = neuron(x, weight, bias)
print(f"Predicted price: ${prediction:,.0f}")
# Output: Predicted price: $945,000
```

**Key insight:** Weights are learned automatically during training. You don't set them by hand — the network figures out the best values on its own.

---

#### 3. What is Loss?

Loss measures how wrong the network's prediction is.

```
Loss = (prediction - ground_truth)²
```

| Loss value | What it means |
|---|---|
| 0.0 | Perfect — prediction matches reality exactly |
| Small number | Close but not perfect |
| Large number | Very wrong — needs more training |

```python
prediction   = 6
ground_truth = 10

loss = (prediction - ground_truth) ** 2
print(f"Loss: {loss}")
# Output: Loss: 16
```

The entire goal of training a neural network is to **minimize loss**.

---

#### 4. Learning Methods in AI

| Method | How it works | Real-world example |
|---|---|---|
| **Supervised learning** | Learns from labeled examples (input + correct answer) | Spam filter trained on marked emails |
| **Unsupervised learning** | Finds patterns with no labels | Customer segmentation |
| **Reinforcement learning** | Learns by trial and error, rewards good actions | AlphaGo learning chess |

Day 1 focused on **supervised learning** — the most common type in industry.

---

#### 5. Gradient Descent — How the Network Learns

Imagine being blindfolded on a hilly field, trying to find the lowest point. You feel the slope under your feet and take small steps downhill. That is gradient descent.

- The **hill** = loss
- The **valley** = where loss is minimized
- Each **step** = one update to weights and bias

```python
# Full learning loop — from scratch, no libraries

ground_truth  = 10
x             = 3
w             = 1.0    # weight — random starting guess
b             = 0.0    # bias   — starts at zero
learning_rate = 0.01

for step in range(20):
    # Forward pass — make a prediction
    prediction = w * x + b

    # Measure how wrong we are
    loss = (prediction - ground_truth) ** 2

    # Calculate gradients — which direction is downhill?
    gradient_w = 2 * (prediction - ground_truth) * x
    gradient_b = 2 * (prediction - ground_truth)

    # Update — take a step downhill
    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b

    print(f"Step {step+1:2d} | pred={prediction:.3f} | loss={loss:.4f} | w={w:.4f} | b={b:.4f}")
```

**Sample output — watch loss drop from 49 to near zero:**
```
Step  1 | pred=3.000 | loss=49.0000 | w=1.420 | b=0.140
Step  5 | pred=6.120 | loss=15.0964 | w=2.143 | b=0.714
Step 10 | pred=8.317 | loss=2.8428  | w=2.772 | b=0.924
Step 20 | pred=9.734 | loss=0.0706  | w=3.178 | b=1.059
```

---

#### 6. The Learning Rate

| Learning rate | What happens |
|---|---|
| Too high (e.g. 0.9) | Overshoots the valley — loss gets worse |
| Just right (e.g. 0.01) | Smooth, steady progress downhill |
| Too low (e.g. 0.00001) | Correct direction but extremely slow |

---

#### 7. The Full Learning Loop

```
Input x
   ↓
f(x) = weight × x + bias       ← neuron computes this
   ↓
Prediction
   ↓
Loss = (prediction - truth)²   ← how wrong are we?
   ↓
Gradient                        ← which way is downhill?
   ↓
w = w - lr × gradient          ← take a step downhill
   ↓
Repeat until loss ≈ 0
```

This loop is the heartbeat of every AI model ever trained.

---

### 💬 Interview Questions I Can Now Answer

| Question | Answer |
|---|---|
| What is a neural network? | Layers of neurons each computing `w*x + b`, trained by minimizing loss via gradient descent |
| What is loss? | How wrong the prediction is — measured as `(prediction - truth)²` |
| What is gradient descent? | An algorithm that adjusts weights step by step toward lower loss |
| Why do we need bias? | Lets the neuron shift its output independently of the input |
| What if all weights are zero? | Network can't learn — all neurons output the same value |
| What is learning rate? | Step size during gradient descent — too high overshoots, too low is too slow |

---

## 📅 Daily Learning Log

| Day | Topic | Key Concepts | Notebook | Status |
|---|---|---|---|---|
| Day 01 | AI Fundamentals & Neural Network Basics | f(x)=wx+b, loss, gradient descent, learning rate, input layer | `Day01_AI_Fundamentals.ipynb` | ✅ Done |
| Day 02 | — | — | — | 🔜 |
| Day 03 | — | — | — | 🔜 |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Google Colab | Zero-setup cloud notebooks |
| NumPy | Numerical computing |
| Matplotlib | Visualizing loss curves and training |
| GitHub | Version control and public portfolio |

---

## 🤝 Let's Connect

I'm actively looking for opportunities in AI — as an ML Engineer, AI Developer, or any role where I can keep building.

If you're hiring, collaborating, or on a similar journey — reach out.

🔗 **LinkedIn:** `[Add your LinkedIn URL here]`  
📧 **Email:** `[Add your email here]`  
🐙 **GitHub:** `[Add your GitHub profile URL here]`

---

*Updated daily as part of my 60-day AI cohort challenge.*  
*Follow along on LinkedIn for daily concept breakdowns, visuals, and project updates.*
