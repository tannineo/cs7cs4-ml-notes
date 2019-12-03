# week 01

Introduction to the module

## Applications of machine learning

- Artificial photo and video creation
  - https://www.youtube.com/watch?v=3AIpPlzM_qs
  - https://www.youtube.com/watch?v=PCBTZh41Ris
- Customer Service / Chat Robots
  - https://ai.googleblog.com/2018/05/duplex-ai-system-for-natural-conversation.html
- Optical Character Recognition (OCR)
- Spam Detection
- Fraud Detection
- Recommendation / Personalisation
- Information Extraction
- Machine Translation
- Autonomous Driving
- Speech Recognition
- Face Recognition
- 3D Face Reconstruction
  - https://cvl-demos.cs.nott.ac.uk/vrn/
  - http://aaronsplace.co.uk/papers/jackson2017recon/
- Medical Diagnosis
- Employee Access Rights
- Cropping Images
  - https://www.dpreview.com/news/6095193348/twitter-is-using-ai-to-intelligently-crop-photos-around-the-eye-catching-bits
- Draw Landscapes
  - https://www.youtube.com/watch?v=p5U4NgVGAwg
- Animate Still Images
  - http://www.dpreview.com/news/6843849919/this-ai-that-can-generate-a-3d-walking-model-from-a-single-still-image-or-painting
- Assigning Tasks to Employees
- Compose Music
  - example from [avia.ai](https://www.aiva.ai/): https://soundcloud.com/user-95265362/sets/genesis
- Create and Extend Art
- Taking Care of Elderly
- Satisfy Sexual Needs
- Predict Personality/Traits/Preferences/
- Kill Enemies
- Object Recognition
  - https://www.youtube.com/watch?v=EJy0EI3hfSg
  - https://experiments.withgoogle.com/ai/giorgio-cam/view/
  - https://experiments.withgoogle.com/ai/giorgio-cam

## Recommended literatures

Books:

- Aurélien Géron, Hands on Machine – Learning with scikit-learn and Tensorflow (O’Reilly Media, 2017)
- Henrik Brink, Joseph Richards, and Mark Fetherolf, Real-world machine learning (Manning Publications Co., 2016)
- John Paul Mueller and Luca Massaron, – Machine Learning for Dummies (John Wiley & Sons, 2016)
- Michael Nielsen, Neural Networks and – Deep Learning
  - http://neuralnetworksanddeeplearning.com/
- Machine Learning Yearning, Andrew Ng, Book In Progress
  - http://www.mlyearning.org/

Online Courses:

- https://www.udemy.com/course/machinelearning/
- https://www.coursera.org/learn/machine-learning

Other online resources:

- Kaggle
- KDNuggets
- Google AI Blog
- Vaultanalytics
  - https://vaultanalytics.com/ai-applications/
- O‘Reilly AI Newsletter
  - https://www.oreilly.com/ai/newsletter.html
- TechCrunch

## Interesting tools

[Menti](www.menti.com) - the realtime polling website.
[TabNine](https://tabnine.com/) - the AI auto-completion tool (VSCode plugin available out of box)

## Machine Learning Examples

### Regression

classic algorithms:

- linear regression
- K-Nearest Neighbour algorithm

loss functions are needed as the metrics to the models

- Root Mean Square Error (RMSE)

### Classification

- if-else statements
- ML solution - hill climbing

## Machine Learning Classifications

### Supervised Learning

- Typical Tasks:
  - Prediction / Regression
  - Classification
- Typical Algorithms:
  - K-Nearest Neighbours
  - Regression
  - Support Vector Machine
  - Decision Trees and Random Forests
- Training Input:
- Labelled data / training data
  - Instances
  - data points with known values for (independent) variables
  - features
  - attributes and known dependent variables
  - response
  - target
  - label
- Production
  - Input: new instance with know independent variables
  - Output: Predict target / label

### Unsupervised Learning

- Unlabelled Data:
  - Work on the current data instead of predicting for future unknown instances
- Typical Tasks:
  - Clustering (group instances into previously unknown groups)
  - Dimensionality Reduction
- Algorithms
  - Clustering
    - K-means
    - Hierarchical Cluster Analysis (HCA)
    - Expected Maximization
  - Dimensionality Reduction
    - Principal Component Analysis
    - Kernal PCA

### Semi Supervised

- Some labelled but mostly unlabelled data
- For instance, Google Photos:
  - Idenfication of the same persons (unsupervised)
  - Manually adding labels to clusters / persons (supervised)
- Often based on:
  - Deep Belief Networks (DBN)
  - Restricted Boltzmann Machines (RBM)

### Reinforcement learning

An agent observes an environment,
chooses between actions and gets
rewards (or penalties) in return. The
agent learns the best strategy
("policy") for what to do in a given
situation.

Often used for making robots to
learn how to walk; also for
DeepMind‘s AlphaGo

example: https://www.youtube.com/watch?v=V1eYniJ0Rnk

### Other Views

- Instance Based vs. Model Based
  - No training (eg. k-nearest, kernel machines, RBF networks)
  - Model is learned through training data
- Parametric vs. Non-Parametric Models
  - eg. linear regression
  - eg. decision tree

The five tribes of machine learning, Webinar by Pedro Domingos

> https://learning.acm.org/techtalks/machinelearning

Pedro Domingos, The master algorithm: How the quest for the ultimate learning machine will remake our world (Basic Books, 2015)

> http://www.idi.ntnu.no/emner/tdt4173/papers/Domingos-SVM-NN-CBR.pdf

See also: John Paul Mueller and Luca Massaron, Machine Learning for Dummies (John Wiley & Sons, 2016)

| Tribe          | Origins              | Problem               | Solition / Master Algorithm |
| :------------- | :------------------- | :-------------------- | :-------------------------- |
| Symbolists     | Logic, Philosophy    | Knowledge composition | Inverse deduction           |
| Connectionists | Neuroscience         | Credit Assignment     | Backpropagation             |
| Evolutionaries | Evolutionary Biology | Structure Discovery   | Genetic programming         |
| Bayesians      | Statistics           | Uncertainty           | Probablistic Inference      |
| Analogizers    | Psychology           | Similarity            | Kernal Machines             |

## Definition of ML

Tom Mitchell, 1997 in A. Géron:

> A computer program is said to:
> learn from experience E
> with respect to some task T
> and some performance measure P
> if its performance on T, as measured by P, improves with experience E.

`Experience` here means `Data`

ML vs. AI vs. DL

```text

AI------------------------------------------------------------------------------------+
| Programs with the ability to learn and reason like humans.                          |
|                                                                                     |
|  ML-----------------------------------------------------------------------------+   |
|  |  Algorithms with the ability to learn without being explicitly programmed.   |   |
|  |                                                                              |   |
|  |  DL------------------------------------------------------------------+       |   |
|  |  |  Subset of machine learning in which artificial neural networks   |       |   |
|  |  |  adapt and learn from vast amounts of data.                       |       |   |
|  |  |                                                                   |       |   |
|  |  +-------------------------------------------------------------------+       |   |
|  +------------------------------------------------------------------------------+   |
+-------------------------------------------------------------------------------------+

```

## Traditional approaches to problem solving

- Rules / Heuristics / Algorithms
- Bruteforce
- Human Labour

Example for Rules: Spem Detection:

- Goal: decide if an email is spam or not? (it's a classification problem)
- Potential Rule: if an email contains the terms "porn, sex, viagra, ..." then it is a spam.
  - Maybe with wildcards, e.g. `p?rn` or `sex*`
- Problem: When new words appear (e.g. `p0rn` instead of `porn`), the rule needs to be adjusted (lots of work)

## The Machine Learning Approach to Problem Solving

T / P / Example
T = Detect Spam Emails (flag new emails as spam/not-spam)
P = Number of correctly classified (not-)spam emails
E = Dataset with emails being classified as spam/not-spam, the more data, the better P becomes

Updating Machine Learning (can be automated)

```text

     +--  Update Data  <--------------- Launch
     |         |                           A
     V         |       Can be automated    |
   Data        |                           |
     |         V                           |
     +->  Train ML algorithm----> Evaluate Solution

```

There is no 'one' algorithm that can do anything.

## The Machine Learning Landscape(s)

- scikit-learn cheatsheet: https://cdn-images-1.medium.com/max/1920/1*kJiLzEawYtmD7t-VQ3AXmw.png
- Microsoft Azure cheatsheet: https://docs.microsoft.com/zh-cn/azure/machine-learning/algorithm-cheat-sheet
- SAS cheatsheet: https://blogs.sas.com/content/subconsciousmusings/2017/04/12/machine-learning-algorithm-use/
- DLib cheatsheet: http://dlib.net/ml_guide.svg

## Strengths and Weaknesses of Machine Learning

### Good at

1. Problems that would require (too) many rules to solve
2. Problems that are too complex to be solved by rules (e.g. handwriting or speech recognition)
3. Problems that have changing environments/data (e.g. spammers who try to cheat the system)
4. Problems where (lots of) data is already available

### Bad at

- Random number generation
- En/Decryption
- Simple operations such as copying data
- Executing programs (following algorithms)

### It's a blackbox (magic)

- It is often not immediately clear why a machine learning system creates the outputs it creates
- Understanding the reasoning behind the output, requires additional analyses
  - sometimes it's very difficult or impossible to fully understand the reasoning
- Legal implications (auto-mobile transport accident)

### No / Difficult Testing

- With traditional approaches, unit testing is easy
- For instance, title detection:
  - detected title for a specific document will not change when e.g. new documents are added to the corpus
