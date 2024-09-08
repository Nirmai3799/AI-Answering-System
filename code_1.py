import fitz
import openai 
import os 
import json 
import requests
import sys 
import getopt

from slack_sdk import WebClient 
from slack_sdk.errors import SlackApiError
from typing import List, Dict, Any 
import os
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, chain
load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")

slack_client = WebClient(token=os.getenv('SLACK_BOT_TOKEN'))

class PDFParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path 
    def extract_text(self):
        from typing_extensions import Concatenate
        # read text from pdf
        pdfreader = PdfReader(self.pdf_path)
        raw_text = ''
        for i, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
                raw_text += content
        return raw_text
    
class TextSplitterAndEmbeddings:
    def __init__(self, raw_text):
        self.raw_text = raw_text 
    
    def text_chunks_and_embed(self):
        # We need to split the text using Character Text Split such that it sshould not increse token size
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
        )
        texts = text_splitter.split_text(self.raw_text)
        embeddings = OpenAIEmbeddings(api_key=api_key)
        document_search = FAISS.from_texts(texts, embeddings)
        return document_search




class QuestionAnsweringAgent:
    def __init__(self, document_text):
        self.document_text = document_text
    
    def query_openai(self, question):
        te = TextSplitterAndEmbeddings(self.document_text)
        doc_search = te.text_chunks_and_embed()
        docs = doc_search.similarity_search(question, k=3)
        # breakpoint()
        context = "\n\n".join([doc.page_content for doc in docs])
        openai.api_key = api_key
        prompt = f"Generate a structured JSON response with the answer to the following question based on the provided document content. If the question is irrelevant or cannot be answered with high confidence, return Data Not Available.\n\nDocument:\n{context}\nQuestion:\n{question}\nJSON Response Format: {{'question': '{question}', 'answer': ''}}"
        result = openai.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
        answer = result.choices[0].message.content
        print(answer)
        answer = answer.replace('JSON Response Format: ', "").strip() 
        import re 
        answer = re.sub(r"'",'"', answer)       
        result = json.loads(answer)
       
        
        return  result['answer']
    
    def answer_questions(self, questions):
        return {q:self.query_openai(q) for q in questions}

class SlackNotifier:
    def __init__(self, slack_channel, slack_token):
        self.slack_channel = slack_channel
        self.slack_token = slack_token

    def post_to_slack(self, message):
        try:
            payload = {"text": json.dumps(message, indent=2)}
            print(payload)
            response = requests.post('https://hooks.slack.com/services/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',
                                     json=payload)
            
            

            return f"Message sent: {response.text}"
        except SlackApiError as e:
            return f"Error sending message to Slack: {e.response['error']}"
        
class AIAnsweringSystem:
    def __init__(self, pdf_path, questions, slack_channel, slack_token):
        self.pdf_parser = PDFParser(pdf_path)
        self.questions = questions
        self.slack_notifier = SlackNotifier(slack_channel, slack_token)
        
    def process_and_notify(self):
        """Process the PDF and post the results to Slack."""
        document_text = self.pdf_parser.extract_text()
        qa_agent = QuestionAnsweringAgent(document_text)
        answers = qa_agent.answer_questions(self.questions)
        # Format answers as a JSON blob
       
        result = self.slack_notifier.post_to_slack(answers)
        print(result)
if __name__=='__main__':
    pdf_path = 'handbook.pdf'
    questions = [
        "What is the name of the company?",
        "What is their vacation policy?",
        "What is the termination policy?",
        "Who is president of India"]
    slack_channel = "#general"
    slack_token = os.getenv('SLACK_BOT_TOKEN')
    ai_system = AIAnsweringSystem(pdf_path, questions, slack_channel, slack_token)
    ai_system.process_and_notify()

         


