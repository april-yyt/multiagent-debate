from langchain.prompts import ChatPromptTemplate

def cot_b(state, llm):
    prompt = ChatPromptTemplate.from_template(
        "Task: {task}\nPlan: {plan}\nInitial reasoning: {cot_a_result}\n\nProvide a more detailed chain of thought reasoning for this task."
    )
    response = llm.invoke(prompt.format_messages(
        task=state["task"],
        plan=state["plan"],
        cot_a_result=state["cot_a_result"]
    ))
    state["cot_b_result"] = response.content
    return state
