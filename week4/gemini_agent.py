import ast
import json
import math
import operator
import os

import tavily
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize API clients
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment or .env file.")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set in environment or .env file.")

groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
MODEL_NAME = "openai/gpt-oss-120b"

# ============================================================================
# TOOL 1: Web Search using Tavily
# ============================================================================
def web_search(query: str) -> str:
    """Searches the web using Tavily API and returns relevant factual results."""
    try:
        response = tavily_client.search(query=query, include_answer=True, max_results=4)
        
        # If Tavily synthesized direct answer, use it with snippets
        answer = response.get("answer")
        results = response.get("results", [])
        
        snippets = []
        for item in results:
            title = item.get("title", "")
            content = item.get("content", "")
            if content:
                snippets.append(f"- {title}: {content}")
        
        snippets_text = "\n".join(snippets)
        if answer and snippets_text:
            return f"Answer: {answer}\n\nSources / Snippets:\n{snippets_text}"
        if answer:
            return f"Answer: {answer}"
        if snippets_text:
            return f"Search Snippets:\n{snippets_text}"
        return "No search results found for the query."
    except (
        tavily.BadRequestError,
        tavily.InvalidAPIKeyError,
        tavily.UsageLimitExceededError,
        tavily.MissingAPIKeyError,
        KeyError,
        ValueError,
        RuntimeError,
        OSError,
    ) as e:
        return f"Error executing web search: {e!s}"


# ============================================================================
# TOOL 2: Basic & Mathematical Calculations
# ============================================================================
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "abs": abs,
    "round": round,
    "pow": pow,
}

ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}

def _eval_ast_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    if isinstance(node, ast.Name):
        if node.id in ALLOWED_CONSTANTS:
            return ALLOWED_CONSTANTS[node.id]
        raise ValueError(f"Unknown variable or constant: {node.id}")
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCTIONS:
            args = [_eval_ast_node(arg) for arg in node.args]
            return ALLOWED_FUNCTIONS[node.func.id](*args)
        raise ValueError("Unsupported function call in calculation.")
    raise TypeError(f"Unsupported syntax element: {type(node).__name__}")

def calculate(expression: str) -> str:
    """Safely evaluates a string mathematical expression and returns the answer."""
    try:
        expr = expression.strip()
        # Parse expression to AST for safe evaluation without raw eval() security risks
        tree = ast.parse(expr, mode="eval")
        result = _eval_ast_node(tree.body)
        return str(result)
    except (
        SyntaxError,
        ValueError,
        TypeError,
        ZeroDivisionError,
        OverflowError,
        ArithmeticError,
        KeyError,
    ) as e:
        return f"Error evaluating expression '{expression}': {e!s}"


# ============================================================================
# Tool definitions in standard function format
# ============================================================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Perform web search to find current information, recent news, real-time events, and general web queries using Tavily.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string to look up on the web."
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate and compute basic or advanced mathematical expressions such as '2*2', '(15 + 25) / 5', '2**10', 'sqrt(144)', etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g. '2*2' or '50 * 12 + 10'."
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

# Map tool names to the actual Python callable functions
available_tools = {
    "web_search": web_search,
    "calculate": calculate,
}

# ============================================================================
# Agent Implementation
# ============================================================================
def run_agent(user_query: str, verbose: bool = True) -> str:
    """
    Takes any input query from the user, uses Groq (openai/gpt-oss-120b)
    to automatically decide whether to call a tool (web_search, calculate)
    or respond directly, and returns the final answer.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intelligent AI agent equipped with two external tools:\n"
                "1. `web_search`: Use this tool whenever you need up-to-date facts, current events, or information from the web.\n"
                "2. `calculate`: Use this tool whenever you need to compute or evaluate any mathematical expressions (like 2*2, percentages, compound arithmetic, etc.).\n"
                "Always use the appropriate tool when required before providing your final answer."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    max_steps = 5  # Guard against infinite tool loops
    for _ in range(max_steps):
        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message

        # Check if the model decided to call any tools
        if response_message.tool_calls:
            # Append model's assistant message with tool calls to conversation history
            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if verbose:
                    print(f"🤖 LLM selected tool: [{function_name}]")
                    print(f"📥 Tool arguments: {tool_args}")

                # Call the corresponding Python function
                tool_function = available_tools.get(function_name)
                if tool_function:
                    tool_result = tool_function(**tool_args)
                else:
                    tool_result = f"Error: Tool '{function_name}' is not recognized."

                if verbose:
                    print(f"📤 Tool result: {tool_result}\n")

                # Append tool execution result back into messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": str(tool_result),
                    }
                )
        else:
            # LLM completed its reasoning and provided the final response
            final_content = response_message.content or ""
            return final_content

    return "Agent reached maximum tool call iterations without finalizing a response."


# ============================================================================
# Main Demonstration
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(f"AI Agent Initialized with model: {MODEL_NAME}")
    print("Available Tools: web_search, calculate")
    print("=" * 60)

    # Test 1: Math calculation
    query_calc = "What is 25 * 14 + 180 / 3?"
    print(f"\nUser Query: {query_calc}")
    answer_calc = run_agent(query_calc)
    print(f"Final Answer:\n{answer_calc}\n")

    # Test 2: Basic expression as specified by user (e.g. 2*2)
    query_basic = "calculate 2*2"
    print(f"\nUser Query: {query_basic}")
    answer_basic = run_agent(query_basic)
    print(f"Final Answer:\n{answer_basic}\n")

    # Test 3: Web Search
    query_search = "What is the latest status of the James Webb Space Telescope mission?"
    print(f"\nUser Query: {query_search}")
    answer_search = run_agent(query_search)
    print(f"Final Answer:\n{answer_search}\n")
