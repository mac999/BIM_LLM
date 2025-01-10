# title: BIM autogen dataset
# description: BIM train dataset generation automatically
# author: taewook kang
# email: laputa99999@gmail.com
# date:
#   2024.6.1
#   2024.7.1
# license: MIT license
# usage:
# 1. pip install the below import packages
# 2. Create OpenAI API and get the API key token 
# 3. Setup OpenAI LLM model name
# 4. Update PDF documents to the input folder
# 5. Run this source code.
# 6. Use the output folder to get the QA dataset
import os, json, PyPDF2, argparse, re
import camelot, fitz 
from pdfminer.high_level import extract_text
from openai import OpenAI
from typing import List
from tqdm import tqdm

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', 
)

template =  {
    "question": " ",
    "vagueness": " ",
    "answer": " "
}

def fix_json (crptd_json):
    messages = [
        {'role': 'system', 'content': f'You are an API that converts the wrongly formatted JSON into a properly fomatted one by following this template : {template}. Answers that are ambiguous, out of context, or lack sufficient information will be marked as vagueness. The vagueness should be 0 or 1. Only respond with the JSON and no additional text. \n.'},
        {'role': 'user', 'content': 'Wrong JSON: ' + crptd_json}
    ]

    response = client.chat.completions.create(
        model='llama3',
        messages=messages,
        max_tokens=2048,
        n=1,
        stop=None,
        temperature=1,
    )

    response_text = response.choices[0].message.content.strip()
    try:
        json_data = json.loads(response_text)
        print(json_data)
        return json_data
    except json.JSONDecodeError:
        print("The JSON is not valid, reformatting again.")
        return []

def generate_questions_answers(text_chunk):
    messages = [
        {'role': 'system', 'content': 'You are an API that converts bodies of text into multiple questions and answers into a JSON format. Each JSON should contain a single question with a single answer and vagueness. The answer that is ambiguous, out of context, or lack sufficient information will be marked as vagueness. The vagueness should be 0 or 1. Only respond with the JSON and no additional text. \n.'},
        {'role': 'user', 'content': 'Text: ' + text_chunk}
    ]

    response = client.chat.completions.create(
        model='llama3',
        messages=messages,
        max_tokens=2048,
        n=1,
        stop=None,
        temperature=0.1,
    )

    response_text = response.choices[0].message.content.strip()
    try:
        json_data = json.loads(response_text)
        print(json_data)
        return json_data
    except json.JSONDecodeError:
        print("Error: Response is not valid JSON.")
        return []

def extract_text_from_pdf(pdf_path, header_threshold=3, footer_threshold=3):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    extracted_text = ""

    # Extract text from the blocks
    for page in doc:
        page_height = page.rect.height
        text_blocks = page.get_text("blocks")

        content_blocks = [block for block in text_blocks if block[1] > header_threshold and block[3] < page_height - footer_threshold]

        page_text = "\n".join([block[4] for block in content_blocks])
        page_text = re.sub(r'(?<!\.)\n(?!$)', ' ', page_text)
        page_text = re.sub(r'[^\x09-\xFE]', '', page_text)
        page_text = re.sub(r'\.{2,}', '.', page_text)
        extracted_text += page_text + "\n"

    doc.close()
    return extracted_text

def process_text(text: str, chunk_size: int = 500) -> List[dict]:
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    all_responses = []
    for index, chunk in tqdm(enumerate(text_chunks), desc="Processing chunks", unit="chunk"):
        response = generate_questions_answers(chunk)
        try:
            if 'question' in response and 'answer' in response:
                all_responses.append({'index': index, 'question': response['question'], 'answer': response['answer'], 'vagueness': response['vagueness']})
            elif isinstance(response, list):
                for i in range(len(response)):
                    if 'question' in response[i] and 'answer' in response[i]:
                        all_responses.append({'index': index, 'question': response[i]['question'], 'answer': response[i]['answer'], 'vagueness': response[i]['vagueness']})
            elif isinstance(response, dict):
                for key in response.keys():
                    if isinstance(response[key], list):
                        for i in range(len(response[key])):
                            if 'question' in response[key][i] and 'answer' in response[key][i]:
                                all_responses.append({'index': index, 'question': response[key][i]['question'], 'answer': response[key][i]['answer'], 'vagueness': response[key][i]['vagueness']})
        except Exception as e:
            print(f"Error: {e}")
            pass
    return all_responses

def main():    
    parser = argparse.ArgumentParser(description="Extract QA text from PDF files")
    parser.add_argument("--input", default='./input', type=str, required=False, help="The input folder containing PDF files.")
    parser.add_argument("--output", default='./output', type=str, required=False, help="The output JSON file to store the processed responses.")
    parser.add_argument("--chunk_size", default=500, type=int, required=False, help="The chunk size for processing text.")

    args = parser.parse_args()
    print(f"Input PDF folder: {args.input}")
    print(f"Output file: {args.output}")
    
    if not os.path.exists(args.input):
        print(f"The folder {args.input} does not exist.")
        return
    
    pdf_files = [f for f in os.listdir(args.input) if f.endswith('.pdf')]        
    for pdf_file in pdf_files:
        input_file = os.path.join(args.input, pdf_file)
        text = extract_text_from_pdf(input_file)
        text_chunk = process_text(text, args.chunk_size)
        responses = {"responses": text_chunk}

        output_fname = os.path.basename(pdf_file).replace('.pdf', '_qa.json')
        output_file = os.path.join(args.output, output_fname)
        with open(output_file, 'a') as f:
            json.dump(responses, f, indent=2)
            print(f"Processed {pdf_file} and saved QA to {output_fname}")

if __name__ == '__main__':
    main()