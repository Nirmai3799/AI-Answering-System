# AI Answering System

This project is an AI-powered answering system that reads questions from a PDF document and uses OpenAI's GPT-3 to answer the questions. The answers are then posted to a Slack channel.

## Features

- Extracts text from PDF files.
- Uses OpenAI's GPT-3 for question-answering.
- Posts the answers to a specified Slack channel.

## Installation

1. Clone the repository:

    bash
    git clone https://github.com/your-username/AI-Answering-System.git
    cd AI-Answering-System
    

2. Create a virtual environment and activate it:

    bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    

3. Install the dependencies:

    bash
    pip install -r requirements.txt
    

4. Set up environment variables:

    - Create a .env file in the root directory of the project.
    - Add your OpenAI and Slack API keys to the .env file:

      bash
      OPENAI_API_KEY=your_openai_api_key
      
      

## Usage

To run the AI Answering System, execute the following command:

```bash
python code_1.py