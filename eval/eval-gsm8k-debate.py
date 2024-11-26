import os
import json
import argparse
import time
import getpass
from tqdm import tqdm
# from promptbench.prompt_engineering.chain_of_thought import ZSCoT, CoT
import promptbench as pb
# from langchain_mistralai import ChatMistralAI
# from langchain_anthropic import ChatAnthropic
from langchain_ollama.llms import OllamaLLM
from agent_debate.fs_debate_rep import DebateFramework


# Constants
MODEL_NAME = "open-mixtral-8x7b"
DATASET_NAME = "gsm8k"
CHECKPOINT_DIR = "checkpoints"
RESULTS_FILE = "debate_results_fs.json"
MAX_NEW_TOKENS = 1024

def load_dataset():
    print("Loading dataset...")
    return pb.DatasetLoader.load_dataset(DATASET_NAME)

def extract_final_answer(text):
    try:
        # First try to find answer with ## marker
        marker = "##"
        marker_index = text.find(marker)
        if marker_index != -1:
            after_marker = text[marker_index + len(marker):].strip()
            first_token = after_marker.split()[0]
            number = ''.join(filter(str.isdigit, first_token))
            if number:
                return number

        # If ## marker not found, try "The answer is X" format
        answer_phrase = "The answer is"
        answer_index = text.lower().find(answer_phrase.lower())
        if answer_index != -1:
            after_phrase = text[answer_index + len(answer_phrase):].strip()
            # Extract first number found after "The answer is"
            number = ''.join(filter(str.isdigit, after_phrase.split()[0]))
            if number:
                return number

        print(f"No answer format found in response: {text}")
        return None
    except Exception as e:
        print(f"Error extracting answer: {str(e)}")
        return None

def evaluate_framework(dataset, num_samples, use_api=True, use_ollama=False):
    results = []
    correct = 0
    total = 0
    max_retries = 20
    base_wait_time = 2
    
    # Initialize all three agents
    if use_ollama:
        agent_a = OllamaLLM(model="mixtral")
        agent_b = OllamaLLM(model="mixtral")
        agent_c = OllamaLLM(model="mixtral")
    elif use_api:
        agent_a = ChatMistralAI(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=MAX_NEW_TOKENS,
            mistral_api_key=os.getenv('MISTRAL_API_KEY')
        )
        agent_b = ChatMistralAI( 
            model=MODEL_NAME,
            temperature=0,
            max_tokens=MAX_NEW_TOKENS,
            mistral_api_key=os.getenv('MISTRAL_API_KEY')
        )
        # ChatAnthropic(
        #     model="claude-3-5-sonnet-20240620",
        #     temperature=0,
        #     max_tokens=MAX_NEW_TOKENS,
        #     anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
        # )
        agent_c = ChatMistralAI( 
            model=MODEL_NAME,
            temperature=0,
            max_tokens=MAX_NEW_TOKENS,
            mistral_api_key=os.getenv('MISTRAL_API_KEY')
        )
    
    # Create debate framework with three agents
    debate = DebateFramework(agent_a, agent_b, agent_c)
    
    # Evaluate samples
    samples = dataset[:num_samples] if num_samples else dataset
    for item in tqdm(samples, desc="Evaluating Debate Framework"):
        try:
            question = item.get('content')
            correct_answer = item.get('label')
            
            if question is None or correct_answer is None:
                print(f"Warning: Skipping invalid item: {item}")
                continue
            
            # Retry logic for rate limit errors
            debate_result = None
            for attempt in range(max_retries):
                try:
                    # Run debate framework
                    debate_result = debate.run_debate(question)
                    
                    if debate_result and debate_result.answer:
                        print(f"Successfully got answer: {debate_result.answer}")
                        break
                    else:
                        print(f"Failed to get answer on attempt {attempt + 1}/{max_retries}")
                        if attempt < max_retries - 1:
                            time.sleep(base_wait_time)
                            continue
                            
                except Exception as e:
                    print(f"Error on attempt {attempt + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(base_wait_time * (2 ** attempt))
                        continue
                    break
            
            if debate_result:
                predicted_answer = extract_final_answer(debate_result.reasoning)
                if predicted_answer:
                    # Compare answers
                    is_correct = predicted_answer == correct_answer
                    if is_correct:
                        correct += 1
                    
                    results.append({
                        "question": question,
                        "correct_answer": correct_answer,
                        "predicted_answer": predicted_answer,
                        "reasoning": debate_result.reasoning,
                        "is_correct": is_correct,
                        "full_response": debate_result.full_response
                    })
                    
                    total += 1
                    if total % 100 == 0:
                        save_checkpoint(results, correct, total)
                else:
                    print(f"Failed to extract answer from debate result")

                    
        except Exception as e:
            print(f"Error processing item: {str(e)}")
            continue
    
    return results, correct, total

def save_checkpoint(results, correct, total):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_debate_{total}.json")
    with open(checkpoint_file, 'w') as f:
        json.dump({
            "results": results,
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total > 0 else 0
        }, f, indent=2)
    print(f"Checkpoint saved: {checkpoint_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Debate Framework on GSM8K dataset")
    parser.add_argument("--use_api", action="store_true", help="Use Mistral API")
    parser.add_argument("--use_ollama", action="store_true", help="Use Ollama local deployment")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    args = parser.parse_args()
    
    if args.use_api and args.use_ollama:
        raise ValueError("Cannot use both API and Ollama at the same time")
    
    # Set up API keys if needed
    if args.use_api:
        if "MISTRAL_API_KEY" not in os.environ:
            os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")
        if "ANTHROPIC_API_KEY" not in os.environ:
            os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter your Anthropic API key: ")
    
    # Load dataset
    dataset = load_dataset()
    
    # Run evaluation
    num_samples = args.num_samples if args.num_samples else len(dataset)
    print(f"Evaluating on {num_samples} samples")
    
    results, correct, total = evaluate_framework(
        dataset, 
        num_samples,
        use_api=args.use_api,
        use_ollama=args.use_ollama
    )
    
    # Calculate final accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"Final Accuracy: {accuracy:.2%}")
    
    # Save final results
    final_results = {
        "results": results,
        "accuracy": accuracy,
        "total_samples": total,
        "correct_answers": correct
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()