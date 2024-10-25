import json
import os
from openai import OpenAI
import argparse
from openai.types.chat import ChatCompletion

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_answer_from_response(response):
    try:
        # Construct a prompt to ask GPT-4 for the final numerical answer
        gpt_prompt = f"Extract only the final numerical answer from this response:\n{response}"
        
        # Make the API call
        chat_completion: ChatCompletion = client.chat.completions.create(
            messages=[{"role": "user", "content": gpt_prompt}],
            model="gpt-4"
        )
        
        # Extract the answer from the API response
        extracted_answer = chat_completion.choices[0].message.content.strip()
        
        return extracted_answer
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return None


def process_results(input_file, output_file):
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Ensure the structure is correct
        if 'CoT' not in data or 'results' not in data['CoT']:
            raise ValueError("Input JSON does not have the expected 'CoT' or 'results' structure")

        results = data['CoT']['results']
        total_questions = len(results)
        correct_answers = 0

        for item in results:
            # Ensure the item is a dictionary and has the required fields
            if isinstance(item, dict):
                if 'full_response' in item and item['full_response']:
                    # Extract the predicted answer from the full response
                    predicted_answer = extract_answer_from_response(item['full_response'])
                    
                    # Update the predicted_answer field
                    item['predicted_answer'] = predicted_answer
                    
                    # Update the is_correct field based on comparison with the correct answer
                    if 'correct_answer' in item and predicted_answer is not None:
                        item['is_correct'] = (predicted_answer == item['correct_answer'])
                        if item['is_correct']:
                            correct_answers += 1
                else:
                    print(f"Skipping item due to missing 'full_response' field: {item}")
            else:
                print(f"Skipping non-dict item: {item}")

        # Recalculate accuracy based on correct answers
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        data['CoT']['accuracy'] = accuracy

        # Save the corrected results into a new file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Processed {total_questions} questions. Accuracy: {accuracy:.4f}")
        print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"Error processing the file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correct JSON result file using GPT-4")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the corrected JSON file")
    
    args = parser.parse_args()
    process_results(args.input_file, args.output_file)