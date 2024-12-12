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
    agent_d_response: str
    agent_e_response: str
    full_response: Dict

class DebateFramework:
    def __init__(self, initial_solver, critic, alternate_solver, second_critic, summarizer, repeats):
        self.agent_a = initial_solver  # Agent A - Initial solver
        self.agent_b = critic  # Agent B - Critic
        self.agent_c = alternate_solver  # Agent C - Alternative solver
        self.agent_d = second_critic  # Agent D - Second critic
        self.agent_e = summarizer  # Agent E - Final summarizer
        self.few_shot_examples = get_prompt(['chain_of_thought', 'gsm8k'])
        self.repeats = repeats

    def get_initial_solution(self, question: str) -> str:
        """Agent A: Generate initial solution."""
        prompt = f"""
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

Examples:
{self.few_shot_examples}

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

    def get_alternate_solution(self, question: str, initial_solution: str, critique: str) -> str:
        """Agent C: Provide alternative solution based on previous discussion."""
        prompt = f"""Math problem: {question}

Initial solution from Agent A:
{initial_solution}

Critique from Agent B:
{critique}

Your task:
Provide a new step-by-step solution incorporating insights from the discussion.
Please output your answer at the end as ##<your answer (arabic numerals)>
"""
        response = self.agent_c.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

    def get_second_critique(self, question: str, initial_solution: str, first_critique: str, alternate_solution: str) -> str:
        """Agent D: Provide second round of critique."""
        prompt = f"""Math problem: {question}

Initial solution from Agent A:
{initial_solution}

First critique from Agent B:
{first_critique}

Alternative solution from Agent C:
{alternate_solution}

Please analyze all previous solutions and provide:
1. A comparison of the different approaches
2. Identification of any remaining errors or misconceptions
3. Your recommended approach to solve this problem
4. Your own final calculation with step-by-step reasoning
"""
        response = self.agent_d.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

    def get_final_summary(self, question: str, all_responses: Dict) -> str:
        """Agent E: Provide final summary and answer."""
        prompt = f"""Math problem: {question}

Initial solution from Agent A:
{all_responses['initial_solution']}

First critique from Agent B:
{all_responses['first_critique']}

Alternative solution from Agent C:
{all_responses['alternate_solution']}

Second critique from Agent D:
{all_responses['second_critique']}

Your tasks:
1. Summarize the key insights from all participants
2. Identify the most reliable solution approach
3. Provide the final authoritative answer
4. Output the final answer as "The answer is <your answer (arabic numerals)>."
"""
        response = self.agent_e.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

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
            first_critique = self.get_critique(question, initial_solution)
            print("\nAgent B Response:")
            print("-" * 80)
            print(first_critique)
            print("-" * 80)
            
            # Initialize variables for repeated rounds
            alternate_solution = None
            second_critique = None
            
            for repeat_num in range(self.repeats):
                print(f"\n=== Repetition {repeat_num + 1}/{self.repeats} ===")
                
                # Get Agent C's alternative solution
                print("\nAgent C (Alternative Solver):")
                alternate_solution = self.get_alternate_solution(question, initial_solution, first_critique)
                print("\nAgent C Response:")
                print("-" * 80)
                print(alternate_solution)
                print("-" * 80)
                
                # Get Agent D's second critique
                print("\nAgent D (Second Critic):")
                second_critique = self.get_second_critique(question, initial_solution, first_critique, alternate_solution)
                print("\nAgent D Response:")
                print("-" * 80)
                print(second_critique)
                print("-" * 80)

            # Get Agent E's final summary
            all_responses = {
                "initial_solution": initial_solution,
                "first_critique": first_critique,
                "alternate_solution": alternate_solution,
                "second_critique": second_critique
            }
            final_response = self.get_final_summary(question, all_responses)
            print("\nAgent E (Final Summarizer):")
            print("-" * 80)
            print(final_response)
            print("-" * 80)

            # Extract final numerical answer
            final_number = self.extract_number(final_response)

            if final_number is None:
                print("\nWARNING: Failed to extract numerical answer")
                return None

            return DebateResult(
                answer=final_number,
                reasoning=final_response,
                agent_a_response=initial_solution,
                agent_b_response=first_critique,
                agent_c_response=alternate_solution,
                agent_d_response=second_critique,
                agent_e_response=final_response,
                full_response={
                    "question": question,
                    "initial_solution": initial_solution,
                    "first_critique": first_critique,
                    "alternate_solution": alternate_solution,
                    "second_critique": second_critique,
                    "final_response": final_response,
                    "final_number": final_number
                }
            )

        except Exception as e:
            print(f"Error in debate process: {e}")
            return None