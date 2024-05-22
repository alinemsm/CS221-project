import anthropic
import os
from dotenv import load_dotenv
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

def generate_questions_and_answers(client, chunk, prompt_template):
    prompt = prompt_template.format(chunk=chunk)
    completion = get_completion(client, prompt)
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

def preprocess_qa_data(file_path):
    data = []

    with open(file_path) as file:
        content = file.read()
        qa_pairs = content.split("\n\nQ: ")[1:]  # Split and remove the first empty question

        for qa_pair in qa_pairs:
            parts = qa_pair.split("\nA: ", maxsplit=1)
            if len(parts) == 2:
                question, answer = parts
                instruction = question.strip()
                response = answer.strip()
                template = "Instruction:\n{}\n\nResponse:\n{}"
                data.append(template.format(instruction, response))
            else:
                print(f"Skipping malformed question-answer pair: {qa_pair}")

    return data