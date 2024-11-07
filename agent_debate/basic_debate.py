# debate_framework.py

from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class DebateResult:
    answer: str
    reasoning: str
    agent_a_response: str
    agent_b_response: str
    full_response: Dict

class DebateFramework:
    def __init__(self, llm, claude):
        self.llm = llm  # Agent A - Initial solver
        self.claude = claude  # Agent B - Critic and final answer

    def get_initial_solution(self, question: str) -> str:
        """Agent A: Generate initial solution."""
        prompt = f"""Math problem: {question}

Provide a step-by-step solution. End with "Therefore, the answer is [NUMBER]."
"""
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

    def get_final_answer(self, question: str, initial_solution: str) -> str:
        """Agent B: Critique and provide final answer."""
        prompt = f"""Math problem: {question}
Initial solution: {initial_solution}

First, critique the initial solution. Is it correct? Could it be improved?
Then, provide your own step-by-step solution if needed.
End by giving ONLY the final numerical answer with no text or symbols.

For example, if the answer is $18, just return:
18
"""
        response = self.claude.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

    def extract_number(self, text: str) -> Optional[str]:
        """Extract numerical answer from text."""
        try:
            # Extract only digits
            number = ''.join(filter(str.isdigit, text.strip()))
            return number if number else None
        except Exception as e:
            print(f"Error extracting number: {e}")
            return None

    def run_debate(self, question: str) -> Optional[DebateResult]:
        """Run the complete debate process."""
        try:
            # Get Agent A's solution
            print("\nAgent A (Initial Solver):")
            initial_solution = self.get_initial_solution(question)
            print(initial_solution)

            # Get Agent B's critique and final answer
            print("\nAgent B (Critic & Final Answer):")
            final_response = self.get_final_answer(question, initial_solution)
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
                agent_b_response=final_response,
                full_response={
                    "question": question,
                    "initial_solution": initial_solution,
                    "final_response": final_response,
                    "final_number": final_number
                }
            )

        except Exception as e:
            print(f"Error in debate process: {e}")
            return None