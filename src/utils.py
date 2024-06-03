import anthropic
import os
from dotenv import load_dotenv
from rouge import Rouge
import re
import json
import pandas as pd
load_dotenv()

def get_completion(prompt, MODEL_NAME = "claude-3-opus-20240229"):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return client.messages.create(
        model=MODEL_NAME,
        max_tokens=int(1.5 * 2048),
        messages=[{
            "role": 'user', "content": prompt
        }]
    ).content

def generate_questions_and_answers(chunk, prompt_template):
    prompt = prompt_template.format(chunk=chunk)
    completion = get_completion(prompt)
    completion_contents = [completion.text for completion in completion]
    output = "\n".join(completion_contents)
    output = output.replace("Question:", "Q:").replace("Answer:", "A:")
    output = output.replace("(blank line)", "").strip()
    return output

def clean_response_file(file_path, output_file_path):
    # Read the contents of the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Initialize an empty list to store the modified lines
    modified_lines = []

    # Iterate over the lines
    i = 0
    while i < len(lines):
        # Check if the line starts with "Here are some questions"
        if lines[i].startswith("Here are some questions"):
            # Skip the current line and the next line (blank line)
            i += 2
        else:
            # Keep the line and add it to the modified_lines list
            modified_lines.append(lines[i])
            i += 1

    # Write the modified content to the output file
    with open(output_file_path, "w") as file:
        file.writelines(modified_lines)

    print(f"File modified successfully. Output saved to {output_file_path}")

def preprocess_qa_data(content):
    data = []
    qa_pairs = content.split("\n\nQ: ")  # Split and remove the first empty question

    for qa_pair in qa_pairs:
        parts = qa_pair.split("\nA: ", maxsplit=1)
        if len(parts) == 2:
            question, answer = parts
            instruction = question.strip()
            response = answer.strip()
            template = "Instruction:\n{}\nResponse:\n{}"
            data.append(template.format(instruction, response))
        else:
            print(f"Skipping malformed question-answer pair")

    return data

# Function to compute ROUGE-L metric
def compute_rouge_l(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]['rouge-l']['f']

def extract_gpt_scores(file_content):
    questions = re.split(r'(Question \d+:)', file_content)[1:]
    data = []

    for i in range(0, len(questions), 2):
        question_number = re.search(r'Question (\d+):', questions[i]).group(1)
        question_text = questions[i + 1]

        answers = re.findall(r'A(\d+):\s*(?:.*\n)*?\s*Total:\s*(\d+(?:\.\d+)?)', question_text)
        answer_scores = {f'A{answer_number} Total Score': float(total_score) for answer_number, total_score in answers}

        superior_tag_match = re.search(r'Superior:\s*(?:(\d+)|Answer\s+(\d+))(?:[^0-9]*)?', question_text, re.IGNORECASE)
        superior_tag = superior_tag_match.group(1) or superior_tag_match.group(2) if superior_tag_match else None

        question_data = {'Question': int(question_number), 'Superior': int(superior_tag) if superior_tag else None}
        question_data.update(answer_scores)
        data.append(question_data)

    return pd.DataFrame(data)

def load_json(path):
    with open(f"{path}.json", "r") as f:
        return json.load(f)
