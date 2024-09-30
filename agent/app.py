from langgraph.graph import Graph
from agent.nodes.planner import planner
from agent.nodes.cot_short import cot_a
from agent.nodes.replanner import replanner
from agent.nodes.trigger_model import trigger_model
from agent.nodes.cot_long import cot_b
from agent.nodes.summarizer import final_answer


def create_workflow(llm, claude):
    # Define the graph
    workflow = Graph()

    # Add nodes (pass llm or claude where required)
    workflow.add_node("planner", lambda state: planner(state, llm))
    workflow.add_node("cot_a", lambda state: cot_a(state, llm))
    workflow.add_node("replanner", lambda state: replanner(state, llm))
    workflow.add_node("trigger_model", trigger_model)
    workflow.add_node("cot_b", lambda state: cot_b(state, llm))
    workflow.add_node("final_answer", lambda state: final_answer(state, claude))

    # Add edges
    workflow.add_edge("planner", "cot_a")
    workflow.add_edge("cot_a", "replanner")
    workflow.add_edge("replanner", "trigger_model")

    # Conditional edge
    workflow.add_conditional_edges(
        "trigger_model",
        lambda x: "cot_b" if x["trigger_cot_b"] else "final_answer"
    )

    workflow.add_edge("cot_b", "final_answer")

    # Set the entry point
    workflow.set_entry_point("planner")

    # Compile the graph
    return workflow.compile()

def run_workflow(app, input_data):
    try:
        final_state = None
        for state in app.stream(input_data):
            if "__end__" not in state:
                final_state = state
                yield state
        return final_state
    except Exception as e:
        print(f"Error in run_workflow: {e}")
        raise
