"""
Restaurant Order Management AI System using LangGraph.
Implements the architecture specified in prompt2.md:
- LLM order extraction & relevance filtering
- Order confirmation against authoritative menu: burger (10), pizza (20), momos (4)
- Available quantity stored as an integer in state
- Handling partial / unavailable dish orders with 3 retry attempts
- Cooking stage with 60% success / 40% failure probability and retry logic
- Serving stage with 70% success / 30% failure probability, recooking fallback, and apology logic
- Complete LangGraph State with annotated messages and retry counters
- Simulation runner for deterministic test cases (TC1, TC2, TC3)
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from typing import Annotated, Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
from groq import Groq
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RestaurantAgent")

# Load environment variables
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# ============================================================================
# 1. AUTHORITATIVE MENU (from prompt2.md lines 74-77)
# ============================================================================

DEFAULT_MENU: Dict[str, int] = {
    "burger": 10,
    "pizza": 20,
    "momos": 4,
}


# ============================================================================
# 2. LANGGRAPH STATE DEFINITION
# ============================================================================

class RestaurantOrderState(TypedDict):
    """
    LangGraph state matching prompt2.md and user specifications:
    - messages: annotated message for llm and user
    - dish_name: dish name as str
    - quantity: quantity as int
    - available_quantity: available quantity as int (per user correction)
    - status: current state status
    - order_retry_attempts: starts at 3
    - cook_retry_attempts: starts at 2
    - serve_retry_attempts: starts at 2
    - final_result: whether the order was complete or not
    - menu: menu dictionary
    - mock_cook_sequence: optional list of booleans for test cases
    - mock_serve_sequence: optional list of booleans for test cases
    """
    messages: Annotated[List[BaseMessage], add_messages]
    dish_name: str
    quantity: int
    available_quantity: int
    status: str
    order_retry_attempts: int
    cook_retry_attempts: int
    serve_retry_attempts: int
    final_result: str
    menu: Optional[Dict[str, int]]
    mock_cook_sequence: Optional[List[bool]]
    mock_serve_sequence: Optional[List[bool]]


def create_initial_state(
    user_message: Optional[str] = None,
    menu: Optional[Dict[str, int]] = None,
    mock_cook_sequence: Optional[List[bool]] = None,
    mock_serve_sequence: Optional[List[bool]] = None,
) -> RestaurantOrderState:
    """Helper to initialize a clean RestaurantOrderState."""
    msgs: List[BaseMessage] = []
    if user_message:
        msgs.append(HumanMessage(content=user_message))

    return {
        "messages": msgs,
        "dish_name": "",
        "quantity": 0,
        "available_quantity": 0,
        "status": "pending",
        "order_retry_attempts": 3,
        "cook_retry_attempts": 2,
        "serve_retry_attempts": 2,
        "final_result": "Pending",
        "menu": menu if menu is not None else dict(DEFAULT_MENU),
        "mock_cook_sequence": list(mock_cook_sequence) if mock_cook_sequence is not None else None,
        "mock_serve_sequence": list(mock_serve_sequence) if mock_serve_sequence is not None else None,
    }


# ============================================================================
# 3. LLM CLIENT & EXTRACTION HELPER
# ============================================================================

def get_groq_client() -> Groq:
    """Returns an authenticated Groq client."""
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_llm_json(prompt: str, system_prompt: str) -> Dict[str, Any]:
    """Call Groq LLM with JSON response format."""
    client = get_groq_client()
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = completion.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return {"error": str(e)}


# ============================================================================
# 4. GRAPH NODES
# ============================================================================

def order_extract_node(state: RestaurantOrderState) -> Dict[str, Any]:
    """
    LLM node:
    - Extracts dish name and quantity from user input.
    - Rejects input unrelated to food ordering (informs user it is a restaurant ordering AI agent).
    - Detects if user is confirming or rejecting a partial order, or cancelling.
    """
    user_msg = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
            user_msg = msg.content
            break

    system_prompt = (
        "You are an AI order extractor for a restaurant.\n"
        "Analyze the customer's input and respond ONLY with a JSON object containing:\n"
        "- is_food_related: boolean (false if asking general knowledge, science, coding, weather, or non-restaurant topics)\n"
        "- is_cancel: boolean (true if user wants to cancel order, stop, or exit)\n"
        "- confirm_partial: boolean (true if user explicitly accepts/confirms the partial available quantity offered previously, e.g., 'yes', 'ok', 'proceed with 2', 'give me what is available')\n"
        "- reject_partial: boolean (true if user refuses/rejects the partial order and wants to order something else or order again, without specifying a valid new dish yet, e.g., 'no', 'I don't want partial', 'let me reorder')\n"
        "- dish_name: string (name of the single dish requested, lowercase, or empty string if none specified)\n"
        "- quantity: integer (number of items requested, default 1 if not specified, 0 if not food-related)\n"
    )

    user_prompt = (
        f"Customer input: '{user_msg}'\n"
        f"Previous status: {state.get('status')}\n"
        f"Previous dish: {state.get('dish_name')}\n"
        f"Available quantity: {state.get('available_quantity')}"
    )

    extracted = call_llm_json(prompt=user_prompt, system_prompt=system_prompt)

    # 1. Check if unrelated
    if not extracted.get("is_food_related", True):
        msg = (
            "I am a restaurant ordering AI agent, not a general purpose LLM. "
            "I can only help you with ordering food from our menu. Please let me know what dish you would like to order."
        )
        return {
            "messages": [AIMessage(content=msg)],
            "status": "unrelated",
        }

    # 2. Check if user wants to cancel
    if extracted.get("is_cancel", False):
        msg = "Your order has been cancelled. Thank you for visiting our restaurant!"
        return {
            "messages": [AIMessage(content=msg)],
            "status": "cancelled",
            "final_result": "Incomplete: Cancelled by user",
        }

    # 3. Check if user confirmed partial order
    if extracted.get("confirm_partial", False):
        confirmed_qty = state.get("available_quantity", 1)
        dish = state.get("dish_name", "dish")
        msg = f"Thank you for confirming. Proceeding with {confirmed_qty} plate(s) of {dish}."
        return {
            "messages": [AIMessage(content=msg)],
            "quantity": confirmed_qty,
            "status": "confirmed",
        }

    # 4. Check if user explicitly rejected partial and did not specify a new dish
    dish = (extracted.get("dish_name") or "").strip().lower()
    if extracted.get("reject_partial", False) and not dish:
        current_retries = state.get("order_retry_attempts", 3) - 1
        if current_retries <= 0:
            msg = (
                "You have declined the partial order and your order attempts limit has been reached. "
                "Order cannot be completed. We apologize for the inconvenience."
            )
            return {
                "messages": [AIMessage(content=msg)],
                "order_retry_attempts": 0,
                "status": "order_exhausted",
                "final_result": "Incomplete: Order retry limit reached",
            }
        msg = f"Understood, we won't proceed with the partial order. What dish would you like to order instead? (Remaining attempts: {current_retries})"
        return {
            "messages": [AIMessage(content=msg)],
            "order_retry_attempts": current_retries,
            "status": "awaiting_user_retry",
        }

    # 5. Normal order extraction (dish name and quantity)
    qty = extracted.get("quantity") or 1
    try:
        qty = int(qty)
    except (ValueError, TypeError):
        qty = 1

    return {
        "dish_name": dish,
        "quantity": qty,
        "status": "extracted",
    }


def order_confirm_node(state: RestaurantOrderState) -> Dict[str, Any]:
    """
    order_confirm node:
    - Reads the authoritative menu.
    - Decides one of 3 cases:
      1. Order is available: status = "confirmed"
      2. Order is partially available: status = "partial"
      3. Order is not available at all: dish not in menu or zero quantity -> status = "not_available"
    - Writes available quantity as an integer in state (0 if dish is not in menu).
    """
    menu = state.get("menu") or DEFAULT_MENU
    dish_requested = state.get("dish_name", "").strip().lower()
    qty_requested = state.get("quantity", 1)

    matched_key = None
    for k in menu.keys():
        if k.lower() == dish_requested or dish_requested in k.lower():
            matched_key = k
            break

    if not matched_key:
        # Dish not in menu -> write 0 as available quantity (int)
        logger.info("Dish '%s' not found in menu", dish_requested)
        return {
            "available_quantity": 0,
            "status": "not_available",
        }

    available_qty = int(menu.get(matched_key, 0))

    if available_qty >= qty_requested and qty_requested > 0:
        logger.info("Order confirmed: %s (Qty: %d, Available: %d)", matched_key, qty_requested, available_qty)
        return {
            "dish_name": matched_key,
            "available_quantity": available_qty,
            "status": "confirmed",
        }
    elif 0 < available_qty < qty_requested:
        logger.info("Order partial: %s (Qty: %d, Available: %d)", matched_key, qty_requested, available_qty)
        return {
            "dish_name": matched_key,
            "available_quantity": available_qty,
            "status": "partial",
        }
    else:
        logger.info("Order not available: %s has 0 available quantity", matched_key)
        return {
            "dish_name": matched_key,
            "available_quantity": 0,
            "status": "not_available",
        }


def handle_order_feedback_node(state: RestaurantOrderState) -> Dict[str, Any]:
    """
    Node called when status is partial or not_available:
    - Decrements order_retry_attempts counter.
    - If retry counter == 0, terminates with apology.
    - Else prompts user to confirm partial quantity or place a new order.
    """
    current_retries = state.get("order_retry_attempts", 3) - 1
    status = state.get("status")
    dish = state.get("dish_name", "dish")
    avail = state.get("available_quantity", 0)

    if current_retries <= 0:
        apology_msg = (
            f"We apologize, but after multiple attempts we could not finalize your order for '{dish}'. "
            "Our order attempts limit has been reached. Please visit us again later."
        )
        return {
            "messages": [AIMessage(content=apology_msg)],
            "order_retry_attempts": 0,
            "status": "order_exhausted",
            "final_result": "Incomplete: Order retry limit reached",
        }

    if status == "partial":
        feedback_msg = (
            f"We only have {avail} plate(s) of '{dish}' available (you requested {state.get('quantity')}). "
            f"Would you like to confirm the partial order of {avail} plate(s), or order another dish? "
            f"(Remaining attempts: {current_retries})"
        )
    else:  # not_available
        feedback_msg = (
            f"Sorry, '{dish}' is currently not available on our menu. "
            f"Please choose another dish. (Remaining attempts: {current_retries})"
        )

    return {
        "messages": [AIMessage(content=feedback_msg)],
        "order_retry_attempts": current_retries,
        "status": "awaiting_user_retry",
    }


def cook_node(state: RestaurantOrderState) -> Dict[str, Any]:
    """
    cook node:
    - Simulates cooking the confirmed dish.
    - 60% chance of success, 40% chance of failure (or mock sequence if provided).
    - If done: status becomes 'ready'.
    - If fail: decrements cook_retry_attempts.
    - If recooking (status was recooking): decrements cook_retry_attempts.
    """
    dish = state.get("dish_name", "dish")
    qty = state.get("quantity", 1)
    status_prev = state.get("status")

    # Determine success
    mock_seq = state.get("mock_cook_sequence")
    if mock_seq is not None and len(mock_seq) > 0:
        success = mock_seq.pop(0)
    else:
        # Probability function: 60% success, 40% failure
        success = random.random() < 0.60

    current_retries = state.get("cook_retry_attempts", 2)

    # If this is a recooking attempt from serve failure, it consumes a cook retry
    if status_prev in ["serve_failed", "recooking"]:
        current_retries -= 1

    if success:
        logger.info("Cooking succeeded for %d x %s (Remaining cook attempts: %d)", qty, dish, current_retries)
        msg = f"Kitchen: Successfully cooked {qty} plate(s) of {dish}. Order is ready to serve!"
        return {
            "messages": [AIMessage(content=msg)],
            "cook_retry_attempts": current_retries,
            "status": "ready",
            "mock_cook_sequence": mock_seq,
        }
    else:
        current_retries -= 1
        logger.warning("Cooking failed for %s. Remaining cook attempts: %d", dish, current_retries)
        msg = f"Kitchen: Cooking attempt for {dish} failed. Remaining cook attempts: {max(0, current_retries)}."
        return {
            "messages": [AIMessage(content=msg)],
            "cook_retry_attempts": current_retries,
            "status": "cook_failed",
            "mock_cook_sequence": mock_seq,
        }


def serve_node(state: RestaurantOrderState) -> Dict[str, Any]:
    """
    serve node:
    - Simulates serving the ready dish.
    - 70% success, 30% failure probability (or mock sequence if provided).
    - If serve succeeds: status = 'completed'.
    - If serve fails: decrements serve_retry_attempts, status = 'serve_failed'.
    """
    dish = state.get("dish_name", "dish")
    qty = state.get("quantity", 1)

    mock_seq = state.get("mock_serve_sequence")
    if mock_seq is not None and len(mock_seq) > 0:
        success = mock_seq.pop(0)
    else:
        # 70% success, 30% failure probability (as specified by user)
        success = random.random() < 0.70

    if success:
        logger.info("Serve succeeded for %d x %s", qty, dish)
        return {
            "status": "completed",
            "mock_serve_sequence": mock_seq,
        }
    else:
        current_retries = state.get("serve_retry_attempts", 2) - 1
        logger.warning("Serve failed for %s. Remaining serve attempts: %d", dish, current_retries)
        msg = f"Serving staff: Service attempt failed. Remaining serve attempts: {max(0, current_retries)}."
        return {
            "messages": [AIMessage(content=msg)],
            "serve_retry_attempts": current_retries,
            "status": "serve_failed",
            "mock_serve_sequence": mock_seq,
        }


def order_completed_node(state: RestaurantOrderState) -> Dict[str, Any]:
    """LLM issues a completion message when serve succeeds."""
    dish = state.get("dish_name", "dish")
    qty = state.get("quantity", 1)
    msg = f"Your order of {qty} plate(s) of '{dish}' is completed and served! Enjoy your meal!"
    return {
        "messages": [AIMessage(content=msg)],
        "status": "completed",
        "final_result": "Order Completed Successfully",
    }


def apology_and_end_node(state: RestaurantOrderState) -> Dict[str, Any]:
    """
    LLM issues an apology to the user and ends the order when cook or serve fails
    and retry attempts are exhausted.
    """
    dish = state.get("dish_name", "dish")
    status = state.get("status")

    if status == "serve_failed":
        if state.get("cook_retry_attempts", 0) <= 0:
            msg = (
                f"We are deeply sorry, but serving your order of '{dish}' failed and our kitchen has exhausted all cooking retry attempts. "
                "Your order cannot be fulfilled. Please accept our sincerest apologies."
            )
            final = "Incomplete: Cook retries exhausted after serve failure"
        else:
            msg = (
                f"We are deeply sorry, but an issue occurred while serving your order of '{dish}', "
                "and serving retries have been exhausted. Please accept our sincerest apologies."
            )
            final = "Incomplete: Serving failed"
    elif status == "cook_failed" or state.get("cook_retry_attempts", 0) <= 0:
        msg = (
            f"We are deeply sorry, but our kitchen could not successfully prepare your '{dish}' "
            "after multiple attempts. Your order has been cancelled and you will not be charged."
        )
        final = "Incomplete: Kitchen cooking failed"
    else:
        msg = f"We apologize, but we could not complete your order of '{dish}' at this time."
        final = "Incomplete: Order failed"

    return {
        "messages": [AIMessage(content=msg)],
        "status": "failed",
        "final_result": final,
    }


# ============================================================================
# 5. CONDITIONAL ROUTING EDGES
# ============================================================================

def route_after_extract(state: RestaurantOrderState) -> str:
    """Route after order_extract."""
    status = state.get("status")
    if status in ["unrelated", "cancelled", "order_exhausted", "awaiting_user_retry"]:
        return END
    if status == "confirmed":
        return "cook"
    return "order_confirm"


def route_after_confirm(state: RestaurantOrderState) -> str:
    """Route after order_confirm."""
    status = state.get("status")
    if status == "confirmed":
        return "cook"
    # Partial or not_available
    return "handle_order_feedback"


def route_after_cook(state: RestaurantOrderState) -> str:
    """
    Route after cook:
    - If status == 'ready': go to serve
    - If status == 'cook_failed':
      - If cook_retry_attempts > 0: retry cook (1 more attempt allowed)
      - If cook_retry_attempts <= 0: apology_and_end
    """
    status = state.get("status")
    if status == "ready":
        return "serve"

    retries = state.get("cook_retry_attempts", 0)
    if retries > 0:
        return "cook"
    return "apology_and_end"


def route_after_serve(state: RestaurantOrderState) -> str:
    """
    Route after serve:
    - If status == 'completed': go to order_completed
    - If serve fails:
      - If serve_retry_attempts > 0 AND cook_retry_attempts > 0:
        call cook so kitchen retries cooking.
      - If cook has exhausted its retry attempt (cook_retry_attempts <= 0):
        apology_and_end
      - If serve has exhausted its retries (serve_retry_attempts <= 0):
        apology_and_end
    """
    status = state.get("status")
    if status == "completed":
        return "order_completed"

    serve_retries = state.get("serve_retry_attempts", 0)
    cook_retries = state.get("cook_retry_attempts", 0)

    if serve_retries > 0 and cook_retries > 0:
        logger.info(
            "Serve failed; retrying cook (Serve retries remaining: %d, Cook retries remaining: %d)",
            serve_retries,
            cook_retries,
        )
        # Update status so cook knows it is recooking
        state["status"] = "recooking"
        return "cook"

    return "apology_and_end"


# ============================================================================
# 6. GRAPH CONSTRUCTION
# ============================================================================

def build_restaurant_graph():
    """Builds and compiles the LangGraph for the Restaurant Order Management System."""
    builder = StateGraph(RestaurantOrderState)

    # Add Nodes
    builder.add_node("order_extract", order_extract_node)
    builder.add_node("order_confirm", order_confirm_node)
    builder.add_node("handle_order_feedback", handle_order_feedback_node)
    builder.add_node("cook", cook_node)
    builder.add_node("serve", serve_node)
    builder.add_node("order_completed", order_completed_node)
    builder.add_node("apology_and_end", apology_and_end_node)

    # Entry point
    builder.set_entry_point("order_extract")

    # Add Conditional Edges
    builder.add_conditional_edges(
        "order_extract",
        route_after_extract,
        {
            "cook": "cook",
            "order_confirm": "order_confirm",
            END: END,
        },
    )

    builder.add_conditional_edges(
        "order_confirm",
        route_after_confirm,
        {
            "cook": "cook",
            "handle_order_feedback": "handle_order_feedback",
        },
    )

    builder.add_edge("handle_order_feedback", END)

    builder.add_conditional_edges(
        "cook",
        route_after_cook,
        {
            "serve": "serve",
            "cook": "cook",
            "apology_and_end": "apology_and_end",
        },
    )

    builder.add_conditional_edges(
        "serve",
        route_after_serve,
        {
            "order_completed": "order_completed",
            "cook": "cook",
            "apology_and_end": "apology_and_end",
        },
    )

    builder.add_edge("order_completed", END)
    builder.add_edge("apology_and_end", END)

    return builder.compile()


# ============================================================================
# 7. INTERACTIVE CLI RUNNER & TEST SIMULATOR
# ============================================================================

def run_interactive():
    """Interactive CLI runner for a real user conversation."""
    graph = build_restaurant_graph()
    state = create_initial_state()

    print("\n" + "=" * 60)
    print("🍽️  WELCOME TO THE AI RESTAURANT ORDER SYSTEM  🍽️")
    print("=" * 60)
    print("Authoritative Menu:")
    for dish, qty in (state.get("menu") or DEFAULT_MENU).items():
        print(f"  - {dish.title()}: {qty} available")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting session. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Session closed.")
            break

        state["messages"].append(HumanMessage(content=user_input))
        result = graph.invoke(state)
        state.update(result)

        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage) or (hasattr(msg, "type") and msg.type == "ai"):
                print(f"\nAI Agent: {msg.content}")

        print(
            f"\n[Status: '{state.get('status')}', Final Result: '{state.get('final_result')}', "
            f"Order Retries: {state.get('order_retry_attempts')}, Cook Retries: {state.get('cook_retry_attempts')}, "
            f"Serve Retries: {state.get('serve_retry_attempts')}, Avail Qty: {state.get('available_quantity')}]\n"
        )

        if state.get("final_result") not in ["Pending", None] and state.get("status") in [
            "completed",
            "failed",
            "cancelled",
            "order_exhausted",
        ]:
            print("=" * 60)
            print(f"🏁 Session Ended: {state.get('final_result')}")
            print("=" * 60)
            break


def simulate_order(
    inputs: List[str],
    menu: Optional[Dict[str, int]] = None,
    mock_cook_sequence: Optional[List[bool]] = None,
    mock_serve_sequence: Optional[List[bool]] = None,
) -> RestaurantOrderState:
    """
    Deterministic/automated simulator for test cases.
    Feeds a list of user inputs turn-by-turn into the graph until completion.
    """
    graph = build_restaurant_graph()
    state = create_initial_state(
        menu=menu,
        mock_cook_sequence=mock_cook_sequence,
        mock_serve_sequence=mock_serve_sequence,
    )

    for idx, user_input in enumerate(inputs):
        logger.info("[Turn %d] Customer: '%s'", idx + 1, user_input)
        state["messages"].append(HumanMessage(content=user_input))
        result = graph.invoke(state)
        state.update(result)
        logger.info(
            "[Turn %d] Status: %s, Final Result: %s, Order Retries: %d, Cook Retries: %d, Serve Retries: %d",
            idx + 1,
            state.get("status"),
            state.get("final_result"),
            state.get("order_retry_attempts"),
            state.get("cook_retry_attempts"),
            state.get("serve_retry_attempts"),
        )

        if state.get("final_result") != "Pending" and state.get("status") in [
            "completed",
            "failed",
            "cancelled",
            "order_exhausted",
        ]:
            break

    return state


if __name__ == "__main__":
    run_interactive()
