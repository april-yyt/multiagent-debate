import os
import json
import re
from glob import glob
from tqdm import tqdm

# Constants
CHECKPOINT_DIR = "checkpoints"
CURATED_DIR = "checkpoints_curated"
RESULTS_FILE = "debate_results_fs.json"
CURATED_RESULTS_FILE = "debate_results_fs_curated.json"

def clean_number_string(text):
    """
    Clean a number string by handling currency and decimal values correctly
    """
    # Remove currency symbols and whitespace
    text = text.replace('$', '').replace(',', '').strip()
    
    # If there's a decimal point, handle it appropriately
    if '.' in text:
        try:
            # Convert to float first to handle decimal points
            value = float(text)
            # If it's a whole number (like 16.00), convert to int
            if value.is_integer():
                return str(int(value))
            # Otherwise keep the decimal value as is
            return str(value)
        except ValueError:
            return text
    
    # If no decimal point, just return digits
    digits = ''.join(c for c in text if c.isdigit())
    return digits if digits else text

def compare_answers(predicted, correct):
    """
    Compare predicted and correct answers after normalizing both
    """
    pred_clean = clean_number_string(str(predicted))
    correct_clean = clean_number_string(str(correct))
    
    if pred_clean != correct_clean:
        print(f"\nMismatch detected:")
        print(f"Original predicted: {predicted} -> Cleaned: {pred_clean}")
        print(f"Original correct: {correct} -> Cleaned: {correct_clean}")
    
    return pred_clean == correct_clean



def extract_final_answer(text):
    """
    Extract the final numerical answer from the reasoning text,
    handling special cases like currency values.
    """
    try:
        # First try to find answer with ## marker
        marker = "##"
        marker_index = text.find(marker)
        if marker_index != -1:
            after_marker = text[marker_index + len(marker):].strip()
            first_token = after_marker.split()[0]
            return clean_number_string(first_token)

        # If ## marker not found, try "The answer is X" format
        patterns = [
            r"The answer is (\$?\d+\.?\d*)",
            r"Therefore, the answer is (\$?\d+\.?\d*)",
            r"answer is (\$?\d+\.?\d*)",
            r"final answer is (\$?\d+\.?\d*)",
            r"equals (\$?\d+\.?\d*)",
            r"= (\$?\d+\.?\d*)",
            r"is (\$?\d+\.?\d*)\.",  # Catches "The final value is $16.00."
            r"(\$?\d+\.?\d*) is correct",
            r"(\$?\d+\.?\d*) would be correct"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the last match if multiple exist
                return clean_number_string(matches[-1])

        # If no patterns match, try to find any number in the last sentence
        sentences = text.split('.')
        if sentences:
            last_sentence = sentences[-1]
            numbers = re.findall(r'\$?\d+\.?\d*', last_sentence)
            if numbers:
                return clean_number_string(numbers[-1])

        print(f"\nNo answer format found in response. Full text:")
        print(f"{text}")
        return None
    except Exception as e:
        print(f"Error extracting answer: {str(e)}")
        print(f"From text: {text}")
        return None

def curate_file(filepath):
    """
    Curate a single checkpoint or results file by only attempting to fix incorrect samples
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    correct = 0
    total = len(results)
    fixed = 0
    
    # Only attempt to fix samples that were originally marked as incorrect
    for i, result in enumerate(results):
        if result.get('is_correct', False):
            # Keep originally correct samples as-is
            correct += 1
            continue
            
        # Only try to extract and compare answers for previously incorrect samples
        if 'reasoning' in result:
            predicted_answer = extract_final_answer(result['reasoning'])
            if predicted_answer:
                result['predicted_answer'] = predicted_answer
                result['is_correct'] = compare_answers(predicted_answer, result['correct_answer'])
                if result['is_correct']:
                    correct += 1
                    fixed += 1
                    print(f"\nFixed sample {i}:")
                    print(f"Original answer: {result.get('predicted_answer', 'N/A')}")
                    print(f"New answer: {predicted_answer}")
                    print(f"Correct answer: {result['correct_answer']}")
    
    # Update accuracy statistics
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nStats for {filepath}:")
    print(f"Total samples: {total}")
    print(f"Total correct: {correct}")
    print(f"Newly fixed samples: {fixed}")
    print(f"Final accuracy: {accuracy:.2%}")
    
    # Compare with original stats
    if 'correct' in data:
        print(f"Original correct count: {data['correct']}")
        print(f"Difference in correct count: {correct - data['correct']}")
    
    # Prepare curated data structure
    curated_data = {
        "results": results,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "fixed_samples": fixed
    }
    
    return curated_data

def main():
    # Create curated directory
    os.makedirs(CURATED_DIR, exist_ok=True)
    
    # Process all checkpoint files
    checkpoint_files = glob(os.path.join(CHECKPOINT_DIR, "checkpoint_debate_*.json"))
    
    print(f"Found {len(checkpoint_files)} checkpoint files to process")
    
    for filepath in tqdm(checkpoint_files, desc="Processing checkpoints"):
        try:
            curated_data = curate_file(filepath)
            
            # Save curated checkpoint
            filename = os.path.basename(filepath)
            curated_filepath = os.path.join(CURATED_DIR, f"curated_{filename}")
            
            with open(curated_filepath, 'w') as f:
                json.dump(curated_data, f, indent=2)
                
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
    
    # Process final results file if it exists
    if os.path.exists(RESULTS_FILE):
        try:
            print("Processing final results file")
            curated_data = curate_file(RESULTS_FILE)
            
            with open(CURATED_RESULTS_FILE, 'w') as f:
                json.dump(curated_data, f, indent=2)
                
            print(f"Final curated accuracy: {curated_data['accuracy']:.2%}")
            
        except Exception as e:
            print(f"Error processing final results: {str(e)}")
    
    print("Curation complete!")
    print(f"Curated checkpoints saved to: {CURATED_DIR}")
    print(f"Curated final results saved to: {CURATED_RESULTS_FILE}")

if __name__ == "__main__":
    main()