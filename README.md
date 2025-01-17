# Chatbot Made Using MyAI
A light tailed chatbot developed using PyTorch capable of answering simple questions and doing basic arithmetic operations like (+, -, /, *).

---
## How It Works

This chatbot deploys a small dataset and a very simple neural network model in order to understand input provided by the user and give meaningful responses.

### Step-by-Step Overview:

1. **Input Handling**
The app receives input from the client, e.g., "Hello there."

2. **Tokenization**
The input sentence is tokenized into individual words, returning an array like:

["Hello", "there"]
``

3. **Word Encoding**
The model compares these tokens against a predefined vocabulary from the training dataset.
- Words in the dataset are converted into a "bag-of-words" representation:

- [1, 0, 0] # 'Hello' matches, others do not
  
- Here, "1" indicates a match and "0" means no match.

4. **Prediction**
The encoded input is run through a trained neural network that predicts the intent of the user with respect to predefined categories like "Greeting," "Goodbye."

5. **Response Generation**
The model picks up an appropriate response from a JSON file. Example,
- **"Greeting"**: "Hello! How can I help you today?"
- **"Goodbye":** "See you next time!"

### Training Data
For training the chatbot, different sample phrases are taken which are categorized by the intent of the phrases. For instance:
- **Greeting:** "Hello, how was your day?"
- **Goodbye:** "See you next time."
It helps the model to learn those patterns and give appropriate responses.

---

## Features

Features

* **Basic Intent Recognition**: It is able to understand and classify the different user intents-Greetings, good-bye, and more.
* **Arithmetic Capabilities**: Performs simple mathematical operations-addition, subtraction, division, and multiplication.
* **Lightweight and Efficient: The system** is lightweight to deploy in cases with relatively small datasets.
 
 
----
 
## How to Run

1. Clone the repository:
   bash
   git clone https://github.com/GameDevRichtofen-G/MyAI-chatbot.git
   cd MyAI-Chatbot
   
   

2. Train the model (optional, if dataset changes):
bash
python train.py


3. Start the chatbot:
bash
python chat.py or py chat.py


4. Interact with the bot!
