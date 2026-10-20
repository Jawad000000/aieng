"""
Restaurant AI Agent using Python, Groq, and LangGraph.
Automates restaurant food ordering, validation, cooking, and serving.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# Ensure stdout handles UTF-8 on Windows
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
# Also search parent directory for .env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# ============================================================================
# 1. AUTHORITATIVE MENU AND INVENTORY
# ============================================================================

DEFAULT_MENU: Dict[str, Dict[str, Any]] = {
    "chicken butter masala": {
        "display_name": "Chicken Butter Masala",
        "price_per_plate": 320.0,
        "available_quantity": 25,
        "is_available": True,
    },
    "chicken biryani": {
        "display_name": "Chicken Biryani",
        "price_per_plate": 280.0,
        "available_quantity": 30,
        "is_available": True,
    },
    "paneer tikka": {
        "display_name": "Paneer Tikka",
        "price_per_plate": 220.0,
        "available_quantity": 15,
        "is_available": True,
    },
    "garlic naan": {
        "display_name": "Garlic Naan",
        "price_per_plate": 60.0,
        "available_quantity": 50,
        "is_available": True,
    },
    "gulab jamun": {
        "display_name": "Gulab Jamun",
        "price_per_plate": 90.0,
        "available_quantity": 0,
        "is_available": False,
    },
}

# In-memory runtime state for menu inventory and orders store
MENU_INVENTORY: Dict[str, Dict[str, Any]] = {}
ORDERS_STORE: Dict[str, Dict[str, Any]] = {}


def reset_store_to_defaults():
    """Resets menu inventory and orders store to clean initial state."""
    global MENU_INVENTORY, ORDERS_STORE
    MENU_INVENTORY.clear()
    for k, v in DEFAULT_MENU.items():
        MENU_INVENTORY[k] = dict(v)
    ORDERS_STORE.clear()


# Initialize on import
reset_store_to_defaults()


def normalize_item_name(name: Optional[str]) -> Optional[str]:
    """Normalizes an item name for case-insensitive lookup."""
    if not name:
        return None
    cleaned = re.sub(r"[^\w\s]", "", name.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Check direct match
    if cleaned in MENU_INVENTORY:
        return cleaned
    # Check partial / substring match
    for key in MENU_INVENTORY:
        if cleaned in key or key in cleaned:
            return key
    return cleaned


def format_menu_for_display() -> str:
    """Formats the current menu and availability for user presentation."""
    lines = ["Here are our available food options:"]
    for key, data in MENU_INVENTORY.items():
        status = (
            f"Available ({data['available_quantity']} plates left)"
            if data["available_quantity"] > 0
            else "Currently Out of Stock"
        )
        lines.append(f"- {data['display_name']}: INR {data['price_per_plate']} per plate [{status}]")
    return "\n".join(lines)


# ============================================================================
# 2. TOOLS
# ============================================================================

def take_order(item_name: str, quantity: int, order_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate and record a customer's food order.
    Authoritative price and inventory come strictly from MENU_INVENTORY.
    """
    normalized = normalize_item_name(item_name)
    
    # 1. Check whether item exists in menu
    if not normalized or normalized not in MENU_INVENTORY:
        logger.warning(f"take_order failed: Unknown item '{item_name}'")
        return {
            "success": False,
            "error_code": "UNKNOWN_ITEM",
            "error_message": f"Sorry, '{item_name}' is not on our menu.",
            "available_quantity": 0,
        }

    item_data = MENU_INVENTORY[normalized]

    # 2. Validate that quantity is a positive integer
    if not isinstance(quantity, int) or quantity <= 0:
        logger.warning(f"take_order failed: Invalid quantity '{quantity}'")
        return {
            "success": False,
            "error_code": "INVALID_QUANTITY",
            "error_message": "Please provide a valid quantity greater than zero.",
            "available_quantity": item_data["available_quantity"],
        }

    # 3. Check whether requested quantity is available
    available = item_data["available_quantity"]
    if quantity > available:
        logger.warning(
            f"take_order failed: Requested {quantity} exceeds available {available} for {item_data['display_name']}"
        )
        return {
            "success": False,
            "error_code": "INSUFFICIENT_INVENTORY",
            "error_message": f"We currently have only {available} plates of {item_data['display_name']} available.",
            "available_quantity": available,
        }

    # 4. Check idempotency / avoid duplicate order creation
    if order_id and order_id in ORDERS_STORE:
        existing = ORDERS_STORE[order_id]
        logger.info(f"take_order: Reusing existing order {order_id}")
        return {
            "success": True,
            "order_id": order_id,
            "item_name": existing["item_name"],
            "confirmed_quantity": existing["quantity"],
            "unit_price": existing["unit_price"],
            "total_price": existing["total_price"],
            "order_status": existing["order_status"],
        }

    # 5. Retrieve authoritative price & calculate total
    unit_price = float(item_data["price_per_plate"])
    total_price = float(quantity * unit_price)

    # 6. Deduct inventory & create order
    item_data["available_quantity"] -= quantity
    if item_data["available_quantity"] == 0:
        item_data["is_available"] = False

    new_order_id = order_id or f"ORD-{uuid.uuid4().hex[:8].upper()}"
    order_record = {
        "order_id": new_order_id,
        "item_key": normalized,
        "item_name": item_data["display_name"],
        "quantity": quantity,
        "unit_price": unit_price,
        "total_price": total_price,
        "order_status": "accepted",
        "cooking_status": "pending",
        "serving_status": "pending",
    }
    ORDERS_STORE[new_order_id] = order_record

    logger.info(
        f"take_order success: {new_order_id} created for {quantity}x {item_data['display_name']} @ {unit_price} each (Total: {total_price})"
    )

    return {
        "success": True,
        "order_id": new_order_id,
        "item_name": item_data["display_name"],
        "confirmed_quantity": quantity,
        "unit_price": unit_price,
        "total_price": total_price,
        "order_status": "accepted",
    }


def cook(order_id: str, simulate_failure: bool = False) -> Dict[str, Any]:
    """
    Simulate cooking an accepted order.
    Requires existing order in 'accepted' status and not already cooked.
    """
    if not order_id or order_id not in ORDERS_STORE:
        logger.error(f"cook failed: Order {order_id} not found")
        return {
            "success": False,
            "error_code": "ORDER_NOT_FOUND",
            "error_message": f"Order {order_id} does not exist.",
            "cooking_status": "failed",
        }

    order = ORDERS_STORE[order_id]

    # Verify order is accepted
    if order["order_status"] != "accepted":
        logger.error(f"cook failed: Order {order_id} is in status '{order['order_status']}', not 'accepted'")
        return {
            "success": False,
            "error_code": "INVALID_ORDER_STATUS",
            "error_message": f"Cannot cook order {order_id}: order status is '{order['order_status']}'.",
            "cooking_status": "failed",
        }

    # Verify not already cooked
    if order["cooking_status"] == "completed":
        logger.warning(f"cook: Order {order_id} has already been cooked.")
        return {
            "success": True,
            "order_id": order_id,
            "cooking_status": "completed",
            "message": f"Order {order_id} was already cooked.",
        }

    if simulate_failure:
        order["cooking_status"] = "failed"
        logger.error(f"cook simulated failure for order {order_id}")
        return {
            "success": False,
            "error_code": "COOKING_FAILED",
            "error_message": f"Kitchen error occurred while preparing order {order_id}.",
            "cooking_status": "failed",
        }

    # Simulate cooking
    order["cooking_status"] = "completed"
    logger.info(f"cook success: Order {order_id} is cooked successfully.")
    return {
        "success": True,
        "order_id": order_id,
        "cooking_status": "completed",
        "message": f"Order {order_id} ({order['quantity']}x {order['item_name']}) cooked successfully.",
    }


def serve(order_id: str, simulate_failure: bool = False) -> Dict[str, Any]:
    """
    Simulate serving or fulfilling a successfully cooked order.
    Requires order to have cooking_status == 'completed' and not already served.
    """
    if not order_id or order_id not in ORDERS_STORE:
        logger.error(f"serve failed: Order {order_id} not found")
        return {
            "success": False,
            "error_code": "ORDER_NOT_FOUND",
            "error_message": f"Order {order_id} does not exist.",
            "serving_status": "failed",
        }

    order = ORDERS_STORE[order_id]

    # Verify order has been cooked successfully
    if order["cooking_status"] != "completed":
        logger.error(f"serve failed: Order {order_id} cooking status is '{order['cooking_status']}'")
        return {
            "success": False,
            "error_code": "NOT_COOKED",
            "error_message": f"Cannot serve order {order_id}: cooking has not completed successfully.",
            "serving_status": "failed",
        }

    # Verify not already served
    if order["serving_status"] == "completed":
        logger.warning(f"serve: Order {order_id} was already served.")
        return {
            "success": True,
            "order_id": order_id,
            "serving_status": "completed",
            "message": f"Order {order_id} was already served.",
        }

    if simulate_failure:
        order["serving_status"] = "failed"
        logger.error(f"serve simulated failure for order {order_id}")
        return {
            "success": False,
            "error_code": "SERVING_FAILED",
            "error_message": f"Fulfillment error occurred while serving order {order_id}.",
            "serving_status": "failed",
        }

    order["serving_status"] = "completed"
    logger.info(f"serve success: Order {order_id} fulfilled and served.")
    return {
        "success": True,
        "order_id": order_id,
        "serving_status": "completed",
        "message": f"Order {order_id} ({order['quantity']}x {order['item_name']}) has been served.",
    }


# ============================================================================
# 3. STATE SCHEMA
# ============================================================================

class RestaurantState(TypedDict, total=False):
    # Chat conversation history
    messages: List[BaseMessage]

    # Natural Language Understanding outputs
    intent: Literal["order", "inquiry", "confirm", "decline", "unrelated", "unknown"]
    raw_user_message: str

    # Extracted slot fields
    item_name: Optional[str]
    quantity: Optional[int]

    # Authoritative financial details (from MENU_INVENTORY, not user input)
    unit_price: Optional[float]
    total_price: Optional[float]
    available_quantity: Optional[int]

    # Confirmation & flow control
    awaiting_confirmation: bool
    customer_confirmation: Optional[bool]
    simulate_cook_failure: bool
    simulate_serve_failure: bool

    # Operational lifecycle tracking
    order_id: Optional[str]
    order_status: Literal["pending", "accepted", "rejected", "failed"]
    cooking_status: Literal["pending", "in_progress", "completed", "failed"]
    serving_status: Literal["pending", "completed", "failed"]

    # Diagnostics & responses
    error: Optional[str]
    final_response: Optional[str]


# ============================================================================
# 4. SYSTEM PROMPT & LLM INTENT EXTRACTION
# ============================================================================

SYSTEM_PROMPT = """You are the intelligent ordering assistant for a premium restaurant.
Your responsibilities:
1. Help customers view the menu, prices, and food availability.
2. Accept food orders by extracting the desired item and quantity.
3. If the user's request is an order:
   - Identify the food item name.
   - Identify the quantity requested as an integer. If the user states zero ("zero" or 0) or a negative number (e.g. -5), you MUST extract that exact integer (0 or -5) so the system can validate and reject it. Only return null if NO quantity was mentioned at all.
   - If the user provides a price, IGNORE it. You are never allowed to confirm or negotiate prices directly; pricing is authoritative from the system.
4. If the user asks general questions about the menu, prices, or availability, classify intent as "inquiry".
5. If the user confirms a pending order (e.g., "yes", "confirm", "looks good", "proceed", "sure", "please place it"), classify intent as "confirm".
6. If the user declines or cancels an order (e.g., "no", "cancel", "decline", "nevermind", "stop"), classify intent as "decline".
7. If the user asks something completely unrelated to the restaurant or food (e.g., weather, politics, math, coding, general trivia), classify intent as "unrelated".
8. Always be polite, professional, and concise.

You MUST respond strictly with a valid JSON object matching this schema:
{
  "intent": "order" | "inquiry" | "confirm" | "decline" | "unrelated",
  "item_name": string or null,
  "quantity": integer or null,
  "user_mentioned_price": number or null,
  "confidence_explanation": string
}
Do not include markdown codeblocks or other formatting outside the JSON object.
"""


def call_llm_for_intent(user_message: str, current_state: RestaurantState) -> Dict[str, Any]:
    """
    Calls Groq LLM to classify user intent and extract slots.
    Falls back gracefully to robust regex heuristic if network / API is unavailable.
    """
    context_item = current_state.get("item_name")
    awaiting_confirm = current_state.get("awaiting_confirmation", False)

    # Fast heuristics for direct confirmation / decline when awaiting confirmation
    msg_clean = user_message.strip().lower()
    if awaiting_confirm:
        if re.match(r"^(yes|yeah|yup|confirm|sure|proceed|ok|okay|please do|go ahead)[\.\!]?$", msg_clean):
            return {
                "intent": "confirm",
                "item_name": context_item,
                "quantity": current_state.get("quantity"),
                "confidence_explanation": "Direct affirmative confirmation",
            }
        if re.match(r"^(no|nope|cancel|decline|dont|stop|nevermind)[\.\!]?$", msg_clean):
            return {
                "intent": "decline",
                "item_name": context_item,
                "quantity": None,
                "confidence_explanation": "Direct cancellation",
            }

    if GROQ_API_KEY:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            prompt_context = f"User input: '{user_message}'\nContext: Currently active item is '{context_item}', awaiting_confirmation={awaiting_confirm}"
            
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_context},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw_text = completion.choices[0].message.content or "{}"
            parsed = json.loads(raw_text)

            # Safety check: if user message explicitly said zero or negative number and LLM returned null
            if parsed.get("quantity") is None:
                if re.search(r"\b(zero|0)\s*(?:plates?|servings?|portions?)?\b", msg_clean):
                    parsed["quantity"] = 0
                else:
                    neg_match = re.search(r"(-\d+)\s*(?:plates?|servings?|portions?)?\b", msg_clean)
                    if neg_match:
                        parsed["quantity"] = int(neg_match.group(1))

            return parsed
        except Exception as e:
            logger.warning(f"Groq API call encountered an error: {e}. Falling back to deterministic NLP extraction.")

    # Deterministic fallback parser for offline/test reliability
    return fallback_intent_extraction(user_message, current_state)


def fallback_intent_extraction(user_message: str, current_state: RestaurantState) -> Dict[str, Any]:
    """Deterministic slot and intent extractor used as backup and test-time guarantee."""
    msg = user_message.lower().strip()
    awaiting = current_state.get("awaiting_confirmation", False)

    # 1. Unrelated queries
    unrelated_triggers = ["weather", "temperature", "who is", "president", "math", "code", "python", "capital of"]
    if any(trigger in msg for trigger in unrelated_triggers):
        return {"intent": "unrelated", "item_name": None, "quantity": None}

    # 2. Confirm / Decline
    if awaiting:
        if any(w in msg for w in ["yes", "confirm", "proceed", "sure", "ok", "place order"]):
            return {"intent": "confirm", "item_name": current_state.get("item_name"), "quantity": current_state.get("quantity")}
        if any(w in msg for w in ["no", "cancel", "decline", "don't", "stop"]):
            return {"intent": "decline", "item_name": None, "quantity": None}

    # 3. Menu / Price Inquiries
    if any(w in msg for w in ["menu", "options", "what do you have", "what is available", "price of", "how much is"]):
        # Check if they are asking about a specific item
        found_item = None
        for key, val in MENU_INVENTORY.items():
            if key in msg or val["display_name"].lower() in msg:
                found_item = val["display_name"]
                break
        return {"intent": "inquiry", "item_name": found_item, "quantity": None}

    # 4. Order slot extraction
    # Look for known item or any item mentioned
    extracted_item = None
    for key, val in MENU_INVENTORY.items():
        if key in msg or val["display_name"].lower() in msg:
            extracted_item = val["display_name"]
            break

    # If no known menu item, check if they specified an unknown food pattern like "want <item>"
    if not extracted_item:
        match_want = re.search(r"(?:want|order|have|give me)\s+(?:(\d+|zero|one|two|three|four|five|ten|twenty)\s+(?:plates?|servings?|bowls?|portions?\s+of\s+)?)?([a-zA-Z\s]+?)(?:\.|$|,|\sat\b)", msg)
        if match_want:
            extracted_item = match_want.group(2).strip()
            # Clean up common trailing words
            extracted_item = re.sub(r"\b(plates?|portions?|bowls?|please)\b", "", extracted_item).strip()

    # Look for quantity
    extracted_qty = None
    # Check for word digits
    word_to_num = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "twenty": 20, "thirty": 30, "fifty": 50}
    for word, num in word_to_num.items():
        if re.search(rf"\b{word}\b", msg):
            extracted_qty = num
            break

    if extracted_qty is None:
        qty_match = re.search(r"(-?\d+)\s*(?:plates?|servings?|bowls?|portions?|x\b)?", msg)
        if qty_match:
            try:
                extracted_qty = int(qty_match.group(1))
            except ValueError:
                pass

    # Carry forward previously identified item if user just provides quantity (Scenario B follow-up)
    if not extracted_item and current_state.get("item_name") and extracted_qty is not None:
        extracted_item = current_state.get("item_name")

    # If the user only said a number or accepted an available quantity suggestion (Scenario D follow-up)
    if not extracted_item and current_state.get("item_name") and (msg == "yes" or "order" in msg):
        extracted_item = current_state.get("item_name")
        if extracted_qty is None and current_state.get("available_quantity") is not None:
            extracted_qty = current_state.get("available_quantity")

    if extracted_item or extracted_qty is not None or any(w in msg for w in ["want", "order", "plates"]):
        return {
            "intent": "order",
            "item_name": extracted_item,
            "quantity": extracted_qty,
        }

    return {"intent": "inquiry", "item_name": None, "quantity": None}


# ============================================================================
# 5. LANGGRAPH NODES
# ============================================================================

def understand_request_node(state: RestaurantState) -> Dict[str, Any]:
    """Node 1: Understand user request and extract slots."""
    messages = state.get("messages", [])
    raw_user_message = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            raw_user_message = m.content
            break
        elif isinstance(m, dict) and m.get("role") == "user":
            raw_user_message = m.get("content", "")
            break

    logger.info(f"understand_request_node: Processing user message: '{raw_user_message}'")
    nlu_result = call_llm_for_intent(raw_user_message, state)
    
    intent = nlu_result.get("intent", "inquiry")
    new_item = nlu_result.get("item_name") or state.get("item_name")
    new_qty = nlu_result.get("quantity")
    if new_qty is None and intent != "order":
        new_qty = state.get("quantity")

    # If customer was awaiting confirmation and said confirm
    if state.get("awaiting_confirmation") and intent == "confirm":
        return {
            "intent": "confirm",
            "raw_user_message": raw_user_message,
            "customer_confirmation": True,
            "awaiting_confirmation": False,
        }

    # If customer was awaiting confirmation and declined
    if state.get("awaiting_confirmation") and intent == "decline":
        return {
            "intent": "decline",
            "raw_user_message": raw_user_message,
            "customer_confirmation": False,
            "awaiting_confirmation": False,
        }

    return {
        "intent": intent,
        "raw_user_message": raw_user_message,
        "item_name": new_item,
        "quantity": new_qty,
    }


def handle_inquiry_or_unrelated_node(state: RestaurantState) -> Dict[str, Any]:
    """Handles unrelated questions or informational restaurant menu inquiries."""
    intent = state.get("intent")
    raw_msg = state.get("raw_user_message", "")
    item_name = state.get("item_name")

    if intent == "unrelated":
        response = (
            "I'm a restaurant ordering assistant. I can help you with our menu, prices, availability, and food orders."
        )
    else:
        # Legitimate inquiry
        normalized = normalize_item_name(item_name)
        if normalized and normalized in MENU_INVENTORY:
            item_data = MENU_INVENTORY[normalized]
            avail = item_data["available_quantity"]
            status_desc = f"{avail} plates available" if avail > 0 else "currently out of stock"
            response = (
                f"{item_data['display_name']} is INR {item_data['price_per_plate']} per plate. "
                f"We currently have {status_desc}."
            )
        else:
            response = format_menu_for_display()

    return {
        "final_response": response,
        "messages": [AIMessage(content=response)],
    }


def validate_order_details_node(state: RestaurantState) -> Dict[str, Any]:
    """Node 2: Validates extracted slots against authoritative menu & inventory."""
    item_name = state.get("item_name")
    quantity = state.get("quantity")

    # Scenario B: Missing quantity
    if not quantity and quantity != 0:
        if item_name:
            norm = normalize_item_name(item_name)
            display = MENU_INVENTORY[norm]["display_name"] if norm in MENU_INVENTORY else item_name
            prompt = f"How many plates of {display} would you like?"
        else:
            prompt = "What item from our menu and how many plates would you like to order?"
        return {
            "error": "MISSING_QUANTITY",
            "final_response": prompt,
            "messages": [AIMessage(content=prompt)],
            "awaiting_confirmation": False,
        }

    # Scenario E: Invalid quantity (zero or negative)
    if quantity <= 0:
        prompt = "Please provide a quantity greater than zero."
        return {
            "error": "INVALID_QUANTITY",
            "final_response": prompt,
            "messages": [AIMessage(content=prompt)],
            "awaiting_confirmation": False,
        }

    # Scenario C: Unknown menu item
    normalized = normalize_item_name(item_name)
    if not normalized or normalized not in MENU_INVENTORY:
        menu_text = format_menu_for_display()
        prompt = f"Sorry, that item isn't on our menu. {menu_text}"
        return {
            "error": "UNKNOWN_ITEM",
            "final_response": prompt,
            "messages": [AIMessage(content=prompt)],
            "awaiting_confirmation": False,
        }

    # Authoritative menu lookups
    item_data = MENU_INVENTORY[normalized]
    display_name = item_data["display_name"]
    unit_price = float(item_data["price_per_plate"])
    available_qty = int(item_data["available_quantity"])

    # Scenario D: Insufficient inventory
    if quantity > available_qty:
        if available_qty > 0:
            prompt = f"We currently have only {available_qty} plates of {display_name} available. Would you like to order {available_qty} plates instead?"
        else:
            prompt = f"Sorry, {display_name} is currently out of stock. Here are our other options:\n{format_menu_for_display()}"
        return {
            "error": "INSUFFICIENT_INVENTORY",
            "available_quantity": available_qty,
            "final_response": prompt,
            "messages": [AIMessage(content=prompt)],
            "awaiting_confirmation": False,
        }

    # Scenario A: Valid order ready for customer confirmation
    total_price = float(quantity * unit_price)
    summary_prompt = (
        f"You are ordering {quantity} plates of {display_name} at INR {unit_price:.2f} per plate. "
        f"The total price is INR {total_price:.2f}. Would you like to confirm this order?"
    )

    return {
        "item_name": display_name,
        "quantity": quantity,
        "unit_price": unit_price,
        "total_price": total_price,
        "available_quantity": available_qty,
        "error": None,
        "awaiting_confirmation": True,
        "final_response": summary_prompt,
        "messages": [AIMessage(content=summary_prompt)],
    }


def handle_decline_node(state: RestaurantState) -> Dict[str, Any]:
    """Scenario G: Customer declined confirmation."""
    response = "No problem! Your order has not been placed. Let me know if you would like anything else."
    return {
        "order_status": "rejected",
        "awaiting_confirmation": False,
        "customer_confirmation": False,
        "final_response": response,
        "messages": [AIMessage(content=response)],
    }


def take_order_node(state: RestaurantState) -> Dict[str, Any]:
    """Node 3: Executes take_order tool after customer confirmation."""
    item_name = state.get("item_name")
    quantity = state.get("quantity")
    existing_order_id = state.get("order_id")

    result = take_order(item_name=item_name, quantity=quantity, order_id=existing_order_id)
    
    if not result.get("success"):
        return {
            "order_status": "failed",
            "error": result.get("error_message", "Order creation failed"),
            "final_response": f"Order creation failed: {result.get('error_message')}",
        }

    return {
        "order_id": result["order_id"],
        "order_status": "accepted",
        "cooking_status": "pending",
        "serving_status": "pending",
        "item_name": result["item_name"],
        "quantity": result["confirmed_quantity"],
        "unit_price": result["unit_price"],
        "total_price": result["total_price"],
        "error": None,
    }


def cook_node(state: RestaurantState) -> Dict[str, Any]:
    """Node 4: Executes cook tool for an accepted order."""
    order_id = state.get("order_id")
    simulate_fail = state.get("simulate_cook_failure", False)

    result = cook(order_id=order_id, simulate_failure=simulate_fail)

    if not result.get("success"):
        return {
            "cooking_status": "failed",
            "error": result.get("error_message", "Cooking failed"),
            "final_response": f"Kitchen error: {result.get('error_message')}. Your order ({order_id}) could not be completed.",
        }

    return {
        "cooking_status": "completed",
        "error": None,
    }


def serve_node(state: RestaurantState) -> Dict[str, Any]:
    """Node 5: Executes serve tool for a cooked order."""
    order_id = state.get("order_id")
    simulate_fail = state.get("simulate_serve_failure", False)

    result = serve(order_id=order_id, simulate_failure=simulate_fail)

    if not result.get("success"):
        return {
            "serving_status": "failed",
            "error": result.get("error_message", "Serving failed"),
            "final_response": f"Fulfillment error: {result.get('error_message')}. Order ({order_id}) could not be served.",
        }

    return {
        "serving_status": "completed",
        "error": None,
    }


def final_response_node(state: RestaurantState) -> Dict[str, Any]:
    """Node 6: Returns the authoritative final confirmation or failure receipt."""
    error = state.get("error")
    order_id = state.get("order_id")
    
    if error or state.get("serving_status") == "failed" or state.get("cooking_status") == "failed" or state.get("order_status") == "failed":
        err_msg = state.get("final_response") or f"Operation could not be completed: {error}"
        return {
            "final_response": err_msg,
            "messages": [AIMessage(content=err_msg)],
        }

    # Successful completion receipt
    item = state.get("item_name")
    qty = state.get("quantity")
    total = state.get("total_price")
    status = state.get("serving_status")

    receipt = (
        f"Order {order_id} Confirmed & Fulfilled!\n"
        f"- Item: {item}\n"
        f"- Quantity: {qty} plates\n"
        f"- Total Price: INR {total:.2f}\n"
        f"- Status: Cooked and Served ({status})\n"
        f"Thank you for dining with us! Enjoy your meal."
    )
    return {
        "final_response": receipt,
        "messages": [AIMessage(content=receipt)],
    }


# ============================================================================
# 6. CONDITIONAL ROUTING & GRAPH CONSTRUCTION
# ============================================================================

def route_after_understanding(state: RestaurantState) -> str:
    """Routes based on classified intent and state flags."""
    intent = state.get("intent")

    if intent in ("unrelated", "inquiry"):
        return "handle_inquiry_or_unrelated"
    elif intent == "decline":
        return "handle_decline"
    elif intent == "confirm":
        # If user explicitly confirmed and we have an order ready
        if state.get("item_name") and state.get("quantity"):
            return "take_order_node"
        return "validate_order_details"
    else:
        # Default to validating the order request
        return "validate_order_details"


def route_after_take_order(state: RestaurantState) -> str:
    """Enforces take_order -> cook, terminating if take_order fails."""
    if state.get("order_status") == "accepted":
        return "cook_node"
    return "final_response_node"


def route_after_cook(state: RestaurantState) -> str:
    """Enforces cook -> serve, terminating if cook fails."""
    if state.get("cooking_status") == "completed":
        return "serve_node"
    return "final_response_node"


def route_after_serve(state: RestaurantState) -> str:
    """Terminates after serve with final response."""
    return "final_response_node"


def build_restaurant_graph(checkpointer: Optional[Any] = None) -> Any:
    """Builds and compiles the deterministic LangGraph workflow."""
    workflow = StateGraph(RestaurantState)

    # Add Nodes
    workflow.add_node("understand_request", understand_request_node)
    workflow.add_node("handle_inquiry_or_unrelated", handle_inquiry_or_unrelated_node)
    workflow.add_node("validate_order_details", validate_order_details_node)
    workflow.add_node("handle_decline", handle_decline_node)
    workflow.add_node("take_order_node", take_order_node)
    workflow.add_node("cook_node", cook_node)
    workflow.add_node("serve_node", serve_node)
    workflow.add_node("final_response_node", final_response_node)

    # Add Edges
    workflow.add_edge(START, "understand_request")

    workflow.add_conditional_edges(
        "understand_request",
        route_after_understanding,
        {
            "handle_inquiry_or_unrelated": "handle_inquiry_or_unrelated",
            "handle_decline": "handle_decline",
            "validate_order_details": "validate_order_details",
            "take_order_node": "take_order_node",
        },
    )

    # Inquiries and declines finish the conversation turn
    workflow.add_edge("handle_inquiry_or_unrelated", END)
    workflow.add_edge("handle_decline", END)

    # Validation finishes the turn so customer can review or clarify
    workflow.add_edge("validate_order_details", END)

    # Order execution pipeline: take_order -> cook -> serve -> final_response
    workflow.add_conditional_edges(
        "take_order_node",
        route_after_take_order,
        {
            "cook_node": "cook_node",
            "final_response_node": "final_response_node",
        },
    )

    workflow.add_conditional_edges(
        "cook_node",
        route_after_cook,
        {
            "serve_node": "serve_node",
            "final_response_node": "final_response_node",
        },
    )

    workflow.add_conditional_edges(
        "serve_node",
        route_after_serve,
        {
            "final_response_node": "final_response_node",
        },
    )

    workflow.add_edge("final_response_node", END)

    # Checkpointer for conversation state persistence
    cp = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=cp)


# Default compiled graph instance
restaurant_agent = build_restaurant_graph()


# ============================================================================
# 7. CONVERSATIONAL RUNNER & CLI INTERACTION
# ============================================================================

def run_agent_interaction(user_input: str, thread_id: str = "default_thread") -> str:
    """Interacts with the agent for a given thread ID."""
    config = {"configurable": {"thread_id": thread_id}}
    input_state = {
        "messages": [HumanMessage(content=user_input)],
    }
    output = restaurant_agent.invoke(input_state, config=config)
    return output.get("final_response", "")


def main():
    """Interactive command-line loop for the Restaurant Agent."""
    print("=" * 65)
    print("  WELCOME TO THE RESTAURANT AI ORDERING ASSISTANT")
    print("=" * 65)
    print("Type your message (or 'exit' to quit).\n")
    
    session_thread = f"session_{uuid.uuid4().hex[:6]}"
    
    while True:
        try:
            user_input = input("Customer: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Thank you for visiting! Goodbye.")
                break

            response = run_agent_interaction(user_input, thread_id=session_thread)
            print(f"\nAssistant: {response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Have a great day!")
            break


if __name__ == "__main__":
    main()
