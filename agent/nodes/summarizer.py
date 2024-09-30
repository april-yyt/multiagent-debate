from langchain.prompts import ChatPromptTemplate

def final_answer(state, claude):
    if "cot_b_result" in state:
        result = state["cot_b_result"]
    else:
        result = state["cot_a_result"]

    prompt = ChatPromptTemplate.from_template(
        "Based on the reasoning: {result}\n\nProvide a concise final answer to the task: {task}"
    )
    response = claude.invoke(prompt.format_messages(task=state["task"], result=result))
    state["final_answer"] = response.content
    print("Final answer state:", state)  # Debug print
    return state
