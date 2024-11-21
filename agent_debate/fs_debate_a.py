from typing import Dict, Optional
from dataclasses import dataclass
from promptbench.prompts.method_oriented import get_prompt

@dataclass
class DebateResult:
    answer: str
    reasoning: str
    agent_a_response: str
    agent_b_response: str
    agent_c_response: str
    full_response: Dict

class DebateFramework:
    def __init__(self, initial_solver, critic, summarizer):
        self.agent_a = initial_solver  # Agent A - Initial solver
        self.agent_b = critic  # Agent B - Critic
        self.agent_c = summarizer  # Agent C - Final summarizer and solver
        self.few_shot_examples = get_prompt(['chain_of_thought', 'gsm8k'])
        
    def extract_number(self, text: str) -> Optional[str]:
        """Extract numerical answer from text."""
        try:
            # Split into lines and look at the last non-empty lines
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            for line in reversed(lines):
                # If line contains only digits (possibly with whitespace)
                if line.replace(' ', '').isdigit():
                    return line.replace(' ', '')
                # If this is a line with digits and other characters
                number = ''.join(filter(str.isdigit, line))
                if number:
                    return number
            return None
        except Exception as e:
            print(f"Error extracting number: {e}")
            return None

    def get_initial_solution(self, question: str) -> str:
        """Agent A: Generate initial solution."""
        prompt = f"""
{self.few_shot_examples}
Q: {question}
Let's think step by step.
Please output your answer at the end as ##<your answer (arabic numerals)>
"""
        print("\nAgent A Prompt:")
        print("-" * 80)
        print(prompt)
        print("-" * 80)

        response = self.agent_a.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

    def get_critique(self, question: str, initial_solution: str) -> str:
        """Agent B: Analyze and critique the initial solution."""
        prompt = f"""Math problem: {question}
Initial solution: {initial_solution}

Follow the [Examples], Please provide a detailed critique of the [Initial Solution]:
1. Is the reasoning correct?
2. Are there any calculation errors?
3. Are there better ways to solve this?
4. What are the key insights that might be missing?

Then, provide your own step-by-step solution.
"""
        print("\nAgent B Prompt:")
        print("-" * 80)
        print(prompt)
        print("-" * 80)

        response = self.agent_b.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    
    def get_final_answer(self, question: str, initial_solution: str, critique: str) -> str:
        """Agent C: Summarize debate and provide final answer."""
        prompt = f"""Math problem: {question}

Initial solution from Agent A:
{initial_solution}

Critique and alternative solution from Agent B:
{critique}

Your tasks:
1. Summarize the key points from both solutions and the critique
2. Provide your own step-by-step solution incorporating the best insights
3. Please output your answer at the end as "The answer is <your answer (arabic numerals)>."

For example:
... your reasoning ...
The answer is <your answer (arabic numerals)>.
"""
        print("\nAgent C Prompt:")
        print("-" * 80)
        print(prompt)
        print("-" * 80)

        response = self.agent_c.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    
    def extract_number(self, text: str) -> Optional[str]:
        """Extract numerical answer from text using the same format as CoT evaluation."""
        try:
            answer_phrase = "The answer is"
            answer_index = text.lower().find(answer_phrase.lower())
            if answer_index != -1:
                after_phrase = text[answer_index + len(answer_phrase):].strip()
                number = ''.join(filter(str.isdigit, after_phrase.split()[0]))
                if number:
                    return number
                
            # Try to find answer with ## marker
            marker = "##"
            marker_index = text.find(marker)
            if marker_index != -1:
                after_marker = text[marker_index + len(marker):].strip()
                first_token = after_marker.split()[0]
                number = ''.join(filter(str.isdigit, first_token))
                if number:
                    return number

            print(f"No answer format found in response: {text}")
            return None
        except Exception as e:
            print(f"Error extracting answer: {e}")
            return None


    def run_debate(self, question: str) -> Optional[DebateResult]:
        """Run the complete debate process."""
        try:
            print("\n" + "="*80)
            print(f"Processing Question: {question}")
            print("="*80 + "\n")

            # Get Agent A's initial solution
            print("\nAgent A (Initial Solver):")
            initial_solution = self.get_initial_solution(question)
            print("\nAgent A Response:")
            print("-" * 80)
            print(initial_solution)
            print("-" * 80)

            # Get Agent B's critique
            print("\nAgent B (Critic):")
            critique = self.get_critique(question, initial_solution)
            print("\nAgent B Response:")
            print("-" * 80)
            print(critique)
            print("-" * 80)

            # Get Agent C's summary and final answer
            print("\nAgent C (Summarizer & Final Solver):")
            final_response = self.get_final_answer(question, initial_solution, critique)
            print("\nAgent C Response:")
            print("-" * 80)
            print(final_response)
            print("-" * 80)

            # Extract final numerical answer
            final_number = self.extract_number(final_response)

            if final_number is None:
                print("\nWARNING: Failed to extract numerical answer")
                print("Last few lines of response:")
                lines = final_response.strip().split('\n')[-5:]
                for line in lines:
                    print(f"Line: '{line}'")
                return None

            print(f"\nExtracted final answer: {final_number}")

            return DebateResult(
                answer=final_number,
                reasoning=final_response,
                agent_a_response=initial_solution,
                agent_b_response=critique,
                agent_c_response=final_response,
                full_response={
                    "question": question,
                    "initial_solution": initial_solution,
                    "critique": critique,
                    "final_response": final_response,
                    "final_number": final_number
                }
            )

        except Exception as e:
            print(f"Error in debate process: {e}")
            return None