# AI-Based Customer Support Chatbot

An AI-based customer support chatbot built using Python, Flask, NLTK, TensorFlow, and Scikit-learn.

The chatbot can understand customer queries, identify the user's intent using a trained neural network model, and return an appropriate response. It also supports basic order-status tracking using order IDs.

## Features

* AI-based customer query classification
* Intent recognition using a trained TensorFlow/Keras model
* Natural Language Processing using NLTK
* Order-status lookup using order IDs
* Flask-based web application
* Interactive web interface
* Confidence-based response handling

## Technologies Used

* Python
* Flask
* TensorFlow / Keras
* NLTK
* NumPy
* Scikit-learn
* HTML/CSS/JavaScript

## Project Structure

```text
CHATBOT/
│
├── app.py
├── train.py
├── intents.json
├── intent_model.h5
├── requirements.txt
├── .gitignore
│
└── templates/
    └── index.html
```

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd CHATBOT
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python app.py
```

### 6. Open the chatbot

Open the following URL in your browser:

```text
http://127.0.0.1:5000
```

## Example Order IDs

The chatbot currently supports the following example order IDs:

* `12345` — Shipped
* `67890` — Out for delivery
* `11111` — Delivered

## Future Improvements

* Connect the chatbot to a real order-management database
* Add authentication for customers
* Improve intent classification with a larger dataset
* Add multilingual support
* Deploy the application to a cloud platform
* Add conversation history and analytics
