from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class DebateResult:
    answer: str
    reasoning: str
    agent_a_response: str
    agent_b_response: str
    agent_c_response: str
    full_response: Dict

class DebateFramework:
    def __init__(self, llm, claude, summarizer):
        self.llm = llm  # Agent A - Initial solver
        self.claude = claude  # Agent B - Critic
        self.summarizer = summarizer  # Agent C - Final summarizer and solver

    def get_initial_solution(self, question: str) -> str:
        """Agent A: Generate initial solution."""
        prompt = f"""Math problem: {question}

Provide a step-by-step solution. End with "Therefore, the answer is [NUMBER]."
"""
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

    def get_critique(self, question: str, initial_solution: str) -> str:
        """Agent B: Analyze and critique the initial solution."""
        prompt = f"""Math problem: {question}
Initial solution: {initial_solution}

Please provide a detailed critique of the initial solution:
1. Is the reasoning correct?
2. Are there any calculation errors?
3. Are there better ways to solve this?
4. What are the key insights that might be missing?

Then, provide your own step-by-step solution if needed.
"""
        response = self.claude.invoke(prompt)
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
3. End with ONLY the final numerical answer with no text or symbols

For example, if the answer is $18, your last line should be just:
18
"""
        response = self.summarizer.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

    def extract_number(self, text: str) -> Optional[str]:
        """Extract numerical answer from text."""
        try:
            # Extract only digits from the last line of the text
            lines = text.strip().split('\n')
            last_line = lines[-1].strip()
            number = ''.join(filter(str.isdigit, last_line))
            return number if number else None
        except Exception as e:
            print(f"Error extracting number: {e}")
            return None

    def run_debate(self, question: str) -> Optional[DebateResult]:
        """Run the complete debate process."""
        try:
            # Get Agent A's initial solution
            print("\nAgent A (Initial Solver):")
            initial_solution = self.get_initial_solution(question)
            print(initial_solution)

            # Get Agent B's critique
            print("\nAgent B (Critic):")
            critique = self.get_critique(question, initial_solution)
            print(critique)

            # Get Agent C's summary and final answer
            print("\nAgent C (Summarizer & Final Solver):")
            final_response = self.get_final_answer(question, initial_solution, critique)
            print(final_response)

            # Extract final numerical answer
            final_number = self.extract_number(final_response)

            if final_number is None:
                print("Failed to extract numerical answer")
                return None

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