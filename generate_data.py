from pypdf import PdfReader
from utils import generate_questions_and_answers, clean_response_file

# Read the PDF file
reader = PdfReader("full_books/casi_clean.pdf")
text = ''.join(page.extract_text() for page in reader.pages)

# Split the text into chunks
chunk_size = 5000
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Read the prompt from prompt.txt
with open("prompt.txt", "r") as file:
    prompt_template = file.read()

# Loop through the chunks and generate questions and answers
for chunk_index, chunk in enumerate(chunks, start=1):
    output = generate_questions_and_answers(chunk, prompt_template)
    with open("response.txt", "a") as file:
        file.write(output + "\n\n")
    print(f"Chunk {chunk_index} processed and saved to response.txt")
