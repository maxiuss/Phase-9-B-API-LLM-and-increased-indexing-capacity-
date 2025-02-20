import json
import openai
import os
import re
from difflib import SequenceMatcher
from semantic_search.semantic_search import semantic_search  # Changed this line - using relative import
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_quiz_from_topic(topic, question_count=3):
    """
    Generate quiz questions by first performing semantic search on the document chunks.
    """
    try:
        # Retrieve relevant text chunks
        print(f"Searching for content about: {topic}")
        search_results = semantic_search(topic, k=3)
        
        if not search_results:
            print("No relevant content found")
            return None
            
        # Combine the text from the results
        relevant_chunks = [result['text'] for result in search_results]
        combined_text = "\n\n".join(relevant_chunks)
        
        print(f"Found {len(relevant_chunks)} relevant chunks")

        # Construct the prompt for the LLM with explicit JSON format instructions
        prompt = (
            f"Generate {question_count} multiple-choice questions based on the following text. "
            "Requirements:\n"
            "1. Options must be complete sentences or valid measurements/ranges\n"
            "2. Numerical answers like '2-5 sets' or '80-92.5% of 1RM' are acceptable\n"
            "3. Never end options with prepositions\n"
            "4. Each option must be unique and meaningful\n"
            "You must return ONLY a JSON array of question objects. Do not wrap the array in any additional object. "
            "The response must start with '[' and end with ']'. "
            "Each question object must follow this exact format:\n"
            "[\n"
            "  {\n"
            '    "question": "Question text here",\n'
            '    "options": {\n'
            '      "A": "First option",\n'
            '      "B": "Second option",\n'
            '      "C": "Third option",\n'
            '      "D": "Fourth option"\n'
            '    },\n'
            '    "correct_answer": "A"\n'
            "  }\n"
            "]\n\n"
            f"Text to base questions on:\n{combined_text}"
        )

        system_content = (
            "You are a specialized quiz generator for strength training content. Follow these rules strictly:\n"
            "1. CRITICAL: Each question MUST use a different answer letter (A,B,C,D)\n"
            "2. Keep options concise but clear (under 15 words per option)\n"
            "3. Include both conceptual and numerical questions\n"
            "4. Base all answers strictly on the provided text\n"
            "5. Make distractors plausible but clearly incorrect\n"
            "6. Use varied formatting for options:\n"
            "   - Short phrases for conceptual questions\n"
            "   - Specific numbers for measurements (e.g., '80-92.5% of 1RM')\n"
            "   - Complete sentences only when necessary\n"
            "7. Randomly distribute correct answers across A, B, C, and D"
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.9  # Increased for more randomness
        )

        quiz_json = normalize_quiz_json(response.choices[0].message["content"])
        
        # Debug output
        print("\nRaw LLM Response:")
        print(quiz_json)
        
        # Clean the JSON string
        quiz_json = quiz_json.strip()
        if quiz_json.startswith("```json"):
            quiz_json = quiz_json.replace("```json", "", 1)
        if quiz_json.endswith("```"):
            quiz_json = quiz_json[:-3]
        quiz_json = quiz_json.strip()

        try:
            quiz = json.loads(quiz_json)
            if not isinstance(quiz, list):
                print("Error: Expected JSON array in response")
                return None
            
            # Validate each question
            valid_questions = []
            for q in quiz:
                is_valid, error = validate_quiz_question(q)
                if is_valid:
                    valid_questions.append(q)
                else:
                    print(f"Skipping invalid question: {error}")
            
            if valid_questions:
                is_valid, message = validate_answer_distribution(valid_questions)
                if not is_valid:
                    print(f"Warning: {message}")
                    
            return valid_questions
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print("Attempted to parse JSON:")
            print(quiz_json)
            return None

    except Exception as e:
        print(f"Error generating quiz: {e}")
        return None

def validate_quiz_question(question):
    """Validate a quiz question for common issues."""
    if not all(key in question for key in ["question", "options", "correct_answer"]):
        return False, "Missing required fields"
        
    options = question["options"]
    
    if len(options) != 4 or not all(opt in options for opt in ['A', 'B', 'C', 'D']):
        return False, "Must have exactly 4 options labeled A, B, C, D"
    
    option_texts = list(options.values())
    
    def is_numeric_option(text):
        """Check if option contains numeric values or measurements."""
        pattern = r'\d+(?:[.-]\d+)?(?:\s*%|\s*RM|\s*sets|\s*reps|\s*pounds?|\s*lbs?|\s*kg)'
        return bool(re.search(pattern, text))
    
    def is_warm_up_sequence(options):
        """Check if options represent a warm-up sequence."""
        weight_patterns = []
        for opt in options:
            matches = re.findall(r'(\d+)\s*(?:pounds?|lbs?)\s*for\s*(\d+)\s*reps?', opt.lower())
            if matches:
                weight_patterns.append(matches[0])
        return len(weight_patterns) >= 3  # True if it's a warm-up sequence
    
    def has_numeric_pattern(text):
        """Check if text contains numeric patterns."""
        patterns = [
            r'\d+(?:[.-]\d+)?(?:\s*%)',  # Percentages
            r'\d+(?:[.-]\d+)?(?:\s*RM)',  # RM values
            r'\d+(?:[.-]\d+)?(?:\s*sets?)',  # Sets
            r'\d+(?:[.-]\d+)?(?:\s*reps?)',  # Reps
            r'\d+(?:[.-]\d+)?(?:\s*(?:pounds?|lbs?|kg))'  # Weights
        ]
        return any(bool(re.search(pattern, text)) for pattern in patterns)
    
    # Skip similarity check for warm-up sequences
    if is_warm_up_sequence(option_texts):
        return True, ""
    
    # Update similarity check for numeric options
    for i, opt1 in enumerate(option_texts):
        for opt2 in option_texts[i+1:]:
            if opt1.lower() == opt2.lower():
                return False, "Contains identical options"
            
            # Skip similarity check if both options have different numeric patterns
            if has_numeric_pattern(opt1) and has_numeric_pattern(opt2):
                if not is_warm_up_sequence(option_texts):
                    # Extract numbers for comparison
                    nums1 = re.findall(r'\d+(?:\.\d+)?', opt1)
                    nums2 = re.findall(r'\d+(?:\.\d+)?', opt2)
                    if nums1 == nums2:
                        return False, "Contains identical numeric values"
                continue
                
            if similarity_ratio(opt1, opt2) > 0.8:
                return False, f"Too similar: '{opt1}' and '{opt2}'"
    
    # Validate other criteria
    prepositions = ['for', 'to', 'in', 'at', 'by', 'of']
    for text in option_texts:
        text = text.strip()
        if len(text) < 10 and not is_numeric_option(text):
            return False, f"Option too short: '{text}'"
        if any(text.lower().endswith(f" {prep}") for prep in prepositions):
            return False, f"Ends with preposition: '{text}'"
    
    return True, ""

def validate_answer_distribution(questions, min_unique_ratio=0.75):
    """Validate answer distribution with strict requirements."""
    if len(questions) <= 1:
        return True, ""
        
    answers = [q["correct_answer"] for q in questions]
    unique_answers = set(answers)
    
    # Reject any quiz with duplicate answers
    if len(unique_answers) != len(questions):
        return False, "Each question must have a different correct answer"
    
    # Check for balanced distribution
    if len(questions) >= 3:
        for letter in 'ABCD':
            if answers.count(letter) > len(questions) // 2:
                return False, f"Answer '{letter}' appears too frequently"
    
    return True, "Good distribution"

def normalize_quiz_json(quiz_json):
    """
    Normalize the JSON structure to ensure it's an array of questions.
    """
    try:
        data = json.loads(quiz_json)
        if isinstance(data, dict) and "questions" in data:
            return json.dumps(data["questions"])
        return quiz_json
    except:
        return quiz_json

def format_quiz_output(quiz):
    """
    Format the quiz questions for display.
    """
    if not quiz:
        return "No quiz generated."
    
    try:
        output = []
        for i, question in enumerate(quiz, 1):
            if not isinstance(question, dict):
                print(f"Warning: Question {i} is not in the expected format")
                continue
                
            output.append(f"\nQuestion {i}:")
            output.append(question.get("question", "Missing question text"))
            options = question.get("options", {})
            for opt, text in options.items():
                output.append(f"{opt}. {text}")
            output.append(f"Correct Answer: {question.get('correct_answer', 'Missing')}")
            output.append("-" * 40)
    
        return "\n".join(output)
    except Exception as e:
        print(f"Error formatting quiz output: {e}")
        return "Error formatting quiz output"

def similarity_ratio(text1, text2):
    """Calculate similarity ratio between two texts."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def compare_numeric_values(text1, text2, tolerance=0.1):
    """Compare numeric values with tolerance for ranges."""
    def extract_numbers(text):
        return [float(n) for n in re.findall(r'\d+(?:\.\d+)?', text)]
    
    nums1 = extract_numbers(text1)
    nums2 = extract_numbers(text2)
    
    if not nums1 or not nums2:
        return False
        
    # Compare ranges with tolerance
    return any(abs(n1 - n2) / max(n1, n2) < tolerance 
              for n1 in nums1 for n2 in nums2)

def classify_question_type(question):
    """Classify question as numerical, conceptual, or technical."""
    options = question["options"]
    
    # Check if any option contains numbers
    has_numbers = any(bool(re.search(r'\d', opt)) for opt in options.values())
    if has_numbers:
        return "numerical"
        
    # Check for technical terms
    technical_terms = ["technique", "form", "position", "stance", "grip"]
    has_technical = any(term in question["question"].lower() for term in technical_terms)
    
    return "technical" if has_technical else "conceptual"

if __name__ == "__main__":
    # Test topics related to your document
    test_topics = [
        "proper lifting technique",
        "warm-up exercises",
        "sets and repetitions"
    ]
    
    print("Testing quiz generation with semantic search...")
    for topic in test_topics:
        print(f"\nGenerating quiz about: {topic}")
        print("=" * 60)
        
        quiz = generate_quiz_from_topic(topic, question_count=2)
        formatted_output = format_quiz_output(quiz)
        print(formatted_output)

