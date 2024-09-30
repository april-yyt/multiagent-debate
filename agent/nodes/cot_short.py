from langchain.prompts import ChatPromptTemplate

def cot_a(state, llm):
    print("CoT A input state:", state)  # Debug print
    prompt = ChatPromptTemplate.from_template(
        "Task: {task}\nPlan: {plan}\n\nProvide a concise chain of thought reasoning for this task."
    )
    response = llm.invoke(prompt.format_messages(task=state["task"], plan=state["plan"]))
    state["cot_a_result"] = response.content
    print("CoT A output state:", state)  # Debug print
    return state
