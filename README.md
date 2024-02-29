# Kaggle Competition
[LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)

# Team Members
Anthony Kwok - [Github](https://github.com/anthonynamnam) | [Kaggle](https://www.kaggle.com/anthonynam)  
Stephen Lee - [Github](https://github.com/sdlee94) | [Kaggle](https://www.kaggle.com/sdlee94)  
Thomas Chong - [Github](https://github.com/thomas-chong) | [Kaggle](https://www.kaggle.com/chongcht710)    
Shreyas Singh - [Github](https://github.com/SinghShreyas) | [Kaggle](https://www.kaggle.com/sinshreyas)   
Srikar Parimi - [Github](https://github.com/MaheshSrikar) | [Kaggle](https://www.kaggle.com/maheshsrikar)    

<hr style="border-top: 3px double #8c8b8b;">

# 🎯 Objectives

## 1. Classify Text as Written by Human vs. AI
- Learning Type: **Supervised**
- Learning Task: **Binary Classification**
- Evaluation Metric: **Area Under the ROC Curve (AUC)**

## 2. Optimize Model for Efficiency
- Efficiency is defined by **(AUC / [BenchmarkAUC - max AUC]) + (RuntimeSeconds / 32400)**  *# somebody please teach me how to display render math notation*
- Must be ranked on the Private Leaderboard higher than the sample_submission.csv benchmark.
- Notebook has to be CPU Only.

<hr style="border-top: 3px double #8c8b8b;">

# 🧠 Approaches

---

## 📊 Training Data
> It is stated that nearly all of the training set were written by students, with only a few generated essays given as examples. A large class imbalance may cause the model to have trouble accurately predicting the minority class.

### Data Augmentation
> The competition states that contestant may generate more essays to use as training data. This would be a form of data augmentation to generate additional samples of the minority class.

---

## 🤖 Modeling
> It is reasonable to assume that Large Language Models (LLMs) are well-suited to our text classification task.

### Choosing the Type of Model
> LLMs can be categorized in into three types of architectures: Encoder-only, Encoder-Decoder and Decoder-only. Since the primary function of the decoder is to generate text, we can focus on **encoder-only** architectures for our text classification task.
Here are several open-source encoder-only LLMs that we can use:

| Name          | Source                         | Number of Parameters |
|---------------|--------------------------------|----------------------|
| BERT-Base     | Google AI Language             | 110 million          |
| BERT-Large    | Google AI Language             | 340 million          |
| RoBERTa-Base  | Facebook AI                    | 125 million          |
| RoBERTa-Large | Facebook AI                    | 355 million          |
| DistilBERT    | Hugging Face                   | 66 million           |
| ALBERT-Base   | Google Research and Toyota TA  | 12 million           |
| ALBERT-Large  | Google Research and Toyota TA  | 18 million           |
| ALBERT-xLarge | Google Research and Toyota TA  | 60 million           |
| ALBERT-xxLarge| Google Research and Toyota TA  | 223 million          |
| Electra-Base  | Google Research                | 110 million          |
| Electra-Large | Google Research                | 335 million          |

### Adapting the Model
> LLMs have a wide-range knowledgebase and understanding of language since they are pre-trained on a massive corpus of text. For this reason, L LMs are often referred to as foundation models. LLMs are known to be good at many different text-based tasks but can fail to perform well in more specialized contexts.

#### Prompt Engineering
> One approach to adapt an LLM for the desired behaviour is prompt engineering (also known as in-context learning). Prompt engineering is the act of formulating the input text (i.e. prompt) in a manner that guides the model towards the desired response.
Prompt engineering is a very accesible and inexpensive way to obtain an LLM with the desired behaviour, but relies on the innate ability of the base model to perform well on the task of interest. We can explore prompt engineering to establish a baseline performance on our classification task.

#### Fine-Tuning
>Fine-tuning LLMs involves the process of further training a pre-trained model on a smaller, specialized dataset to obtain better performance on the task of interest. Fine-tuning is a suitable approach if the neccessary computational resources are available and if the desired performance is unattainable through prompt engineering.

---

## ⬆️ Optimizing Runtime

### Parallel Computing
>Distributing the training across multiple compute nodes in parallel can speed up training.

### Quantization
>Quantization is a process that reduces the precision of the model's weights, typically from 32-bit floating-point to 16-bit floating-point or 8-bit integers. Quantization results in a smaller model size and faster training and inference at some cost to model performance.

### Opting for Efficient Fine-Tuning Approaches
>As opposed to fine-tuning all of a model's weights (full fine-tuning), we can apply Parameter Efficient Fine-Tuning (PEFT) techniques for less computationally intensive model training. The following are the types of PEFT techniques we can look to implement:
- **Selective:** fine-tune subset of initial parameters
- **Reparameterize:** create low-rank representation of initial model weights to reduce amount of parameters to fine-tune (e.g. LoRA)
- **Additive:** add new trainable layers or parameters to model

---

## 1️⃣ Lower-Rank Adaptation (LoRA)

LoRA is one of the PEFT Fine-tuning methods. Instead of modifying the parameters in the Large Language Model **(e.g. 1024 x 1024 = 1,048,576 parameters)**, LoRA add another matrix X with the same size (1024 x 1024) in each layer in the model archtecture and decompose the add-on matrix into two smaller matrix **(e.g. matrix A with 1024 x 8 &  matrix B with 8 x 1024)**. So in total, the number of parameters reduces from 1,048,576 to 16,384, approximately 1.5% of the parameters of the original archtecture.

> Mathematically speaking, You can obtain Matrix X by performing matrix multiplication of **Matrix A** and **Matrix B**.

This is an example to illustrate how to apply LoRA to fine tune a LLM. [Sample Code](https://github.com/anthonynamnam/llm-detect-ai-gen-text/blob/main/lora-fine-tuning-with-distilbert.ipynb)

<hr style="border-top: 3px double #8c8b8b;">