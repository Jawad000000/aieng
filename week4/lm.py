from typing import TypedDict

from langgraph.graph import END, StateGraph

class State(TypedDict):
    number: int

def double(state:State)->dict:
    board_number = state["number"]
    new_number = board_number *2
    print(new_number)
    return {"number": new_number}

def finish(state:State) -> dict:
    board_number = state["number"]
    print("finish")
    print(board_number)
    return {"number": board_number}

def decision(state:State)->str:
    board_number = state["number"]
    if board_number < 100:
        return "double"
    else:
        return "finish"

builder = StateGraph(State)
builder.add_node("double",double)
builder.add_node("finish",finish)

#entry point
builder.set_entry_point("double")

builder.add_conditional_edges(
    "double",
    decision,
    {
        "double":"double",
        "finish":"finish",
    }
)

builder.add_edge("finish",END)

graph=builder.compile()

#test
result=graph.invoke({"number": 5})
print(result)
