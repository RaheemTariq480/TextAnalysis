﻿
# Local LLM-Based Text Analysis Tool

This project uses a locally hosted large language model (LLM) like Ollama to analyze text. The system categorizes the input text, performs sentiment analysis, and detects the emotion conveyed in the text. All processing is done locally for privacy and performance.

## Features
- **Text Categorization**: Classifies the text into user-defined categories.
- **Sentiment Analysis**: Identifies whether the text expresses positive, negative, or neutral sentiment.
- **Emotion Detection**: Detects emotions such as joy, anger, sadness, etc.
- **Local Inference**: Ensures data privacy by running entirely on a local machine.

## Prerequisites
- Python 3.8+
- Git installed

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/RaheemTariq480/TextAnalysis
cd TextAnalysis
```

### 2. Create a Virtual Environment
Create a virtual environment to keep dependencies isolated.

For Linux/MacOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install the Requirements
Install the necessary packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Run the Project
To run the project, execute the main Python script:

```bash
python microservice.py
```

### 5. Usage
Once the project is running, input a text along with predefined categories, and the tool will output the most appropriate category, the sentiment, and the detected emotion.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
