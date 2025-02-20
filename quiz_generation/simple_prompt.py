import os
import openai
import json
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_json(raw_output):
    # Remove markdown code fences by splitting into lines
    lines = raw_output.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    json_clean = "\n".join(lines).strip()

    # If the output starts with a '[' but doesn't end with ']', append a missing bracket
    if json_clean.startswith("[") and not json_clean.endswith("]"):
        json_clean += "]"
    return json_clean

def generate_quiz(text_chunk, question_count=3):
    prompt = (
        f"Generate {question_count} multiple-choice questions based on the text below. "
        "Each question must have a question text, exactly four options labeled A, B, C, D, "
        "and a clearly indicated correct answer. Please return only the results in valid JSON format "
        "without any additional commentary (i.e., provide only a JSON array of objects where each object has keys: "
        "'question', 'options', and 'correct_answer').\n\n"
        f"Text: {text_chunk}"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a quiz generator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    raw_output = response.choices[0].message["content"]
    print("Raw LLM Output:", raw_output)

    try:
        json_str = extract_json(raw_output)
        quiz_output = json.loads(json_str)
        return quiz_output
    except json.JSONDecodeError:
        raise ValueError("LLM response could not be parsed as JSON.")

def validate_quiz(quiz):
    for idx, question in enumerate(quiz):
        # Ensure required keys exist
        if not all(key in question for key in ("question", "options", "correct_answer")):
            raise ValueError(f"Question {idx + 1} is missing required keys.")
        # Check that number of options is within our expected range
        if not (3 <= len(question["options"]) <= 4):
            raise ValueError(f"Question {idx + 1} does not have 3-4 answer options.")
        # Confirm the correct answer key is valid
        if question["correct_answer"] not in question["options"]:
            raise ValueError(f"Question {idx + 1} has an invalid correct answer key.")
    return True

if __name__ == "__main__":
    sample_text = (
        "The theory of natural selection explains how species evolve over time. "
        "Species that adapt successfully to their environment survive and reproduce, "
        "passing on beneficial traits to the next generation."
    )
    
    try:
        quiz = generate_quiz(sample_text, question_count=3)
        # Validate the quiz output before printing
        validate_quiz(quiz)
        
        print("Generated Quiz:")
        for idx, item in enumerate(quiz, start=1):
            print(f"\nQuestion {idx}:")
            print(f"  {item.get('question')}")
            print("  Options:")
            options = item.get('options', {})
            for opt, answer in options.items():
                print(f"    {opt}: {answer}")
            print(f"  Correct Answer: {item.get('correct_answer')}")
    except ValueError as e:
        print("Error generating quiz:", e)




