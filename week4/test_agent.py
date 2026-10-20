"""
Test Suite for Restaurant AI Agent.
Covers all 15 scenarios specified in Section 10 of prompt.md.
"""

import sys
import unittest
import uuid

# Reconfigure encoding for clean Windows output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from restagent import (
    MENU_INVENTORY,
    ORDERS_STORE,
    build_restaurant_graph,
    cook,
    reset_store_to_defaults,
    serve,
    take_order,
)


class TestRestaurantAgent(unittest.TestCase):
    def setUp(self):
        """Reset menu inventory and orders store before each test."""
        reset_store_to_defaults()
        self.checkpointer = MemorySaver()
        self.agent = build_restaurant_graph(checkpointer=self.checkpointer)

    def _send_message(self, text: str, thread_id: str) -> dict:
        """Helper to send a message to the agent and return resulting state."""
        config = {"configurable": {"thread_id": thread_id}}
        return self.agent.invoke(
            {"messages": [HumanMessage(content=text)]},
            config=config,
        )

    # ------------------------------------------------------------------------
    # 1. Valid order with customer confirmation
    # ------------------------------------------------------------------------
    def test_01_valid_order_with_confirmation(self):
        thread_id = f"test_valid_{uuid.uuid4().hex[:6]}"
        
        # Turn 1: Propose valid order
        state_1 = self._send_message("I want 10 plates of Chicken Butter Masala.", thread_id)
        self.assertTrue(state_1.get("awaiting_confirmation"))
        self.assertEqual(state_1.get("item_name"), "Chicken Butter Masala")
        self.assertEqual(state_1.get("quantity"), 10)
        self.assertEqual(state_1.get("unit_price"), 320.0)
        self.assertEqual(state_1.get("total_price"), 3200.0)
        self.assertIn("Would you like to confirm", state_1.get("final_response", ""))

        # Turn 2: Customer confirms
        state_2 = self._send_message("Yes, please confirm the order.", thread_id)
        self.assertEqual(state_2.get("order_status"), "accepted")
        self.assertEqual(state_2.get("cooking_status"), "completed")
        self.assertEqual(state_2.get("serving_status"), "completed")
        self.assertIsNotNone(state_2.get("order_id"))
        self.assertIn("Cooked and Served", state_2.get("final_response", ""))

    # ------------------------------------------------------------------------
    # 2. Missing quantity
    # ------------------------------------------------------------------------
    def test_02_missing_quantity(self):
        thread_id = f"test_missing_qty_{uuid.uuid4().hex[:6]}"
        
        # Turn 1: Item specified without quantity
        state_1 = self._send_message("I want Chicken Biryani.", thread_id)
        self.assertEqual(state_1.get("error"), "MISSING_QUANTITY")
        self.assertIn("How many plates of Chicken Biryani would you like?", state_1.get("final_response", ""))
        self.assertFalse(state_1.get("awaiting_confirmation", False))

        # Turn 2: User provides quantity, preserving context
        state_2 = self._send_message("5 plates please.", thread_id)
        self.assertTrue(state_2.get("awaiting_confirmation"))
        self.assertEqual(state_2.get("quantity"), 5)
        self.assertEqual(state_2.get("total_price"), 5 * 280.0)

    # ------------------------------------------------------------------------
    # 3. Unknown menu item
    # ------------------------------------------------------------------------
    def test_03_unknown_menu_item(self):
        thread_id = f"test_unknown_{uuid.uuid4().hex[:6]}"
        state = self._send_message("I want 2 plates of horse meat.", thread_id)
        self.assertEqual(state.get("error"), "UNKNOWN_ITEM")
        self.assertIn("isn't on our menu", state.get("final_response", ""))
        self.assertIn("Chicken Butter Masala", state.get("final_response", ""))
        # Verify tools cook & serve were NOT called
        self.assertIsNone(state.get("order_id"))
        self.assertIsNone(state.get("cooking_status"))

    # ------------------------------------------------------------------------
    # 4. Zero or negative quantity
    # ------------------------------------------------------------------------
    def test_04_zero_or_negative_quantity(self):
        thread_id = f"test_invalid_qty_{uuid.uuid4().hex[:6]}"
        
        # Zero quantity
        state_zero = self._send_message("I want zero plates of Chicken Biryani.", thread_id)
        self.assertEqual(state_zero.get("error"), "INVALID_QUANTITY")
        self.assertIn("greater than zero", state_zero.get("final_response", ""))
        self.assertIsNone(state_zero.get("order_id"))

        # Negative quantity
        thread_neg = f"test_neg_qty_{uuid.uuid4().hex[:6]}"
        state_neg = self._send_message("I want -5 plates of Chicken Biryani.", thread_neg)
        self.assertEqual(state_neg.get("error"), "INVALID_QUANTITY")
        self.assertIn("greater than zero", state_neg.get("final_response", ""))
        self.assertIsNone(state_neg.get("order_id"))

    # ------------------------------------------------------------------------
    # 5. Quantity exceeding inventory
    # ------------------------------------------------------------------------
    def test_05_quantity_exceeding_inventory(self):
        thread_id = f"test_excess_{uuid.uuid4().hex[:6]}"
        # Chicken Biryani initial stock is 30
        state = self._send_message("I want 50 plates of Chicken Biryani.", thread_id)
        self.assertEqual(state.get("error"), "INSUFFICIENT_INVENTORY")
        self.assertIn("only 30 plates of Chicken Biryani available", state.get("final_response", ""))
        self.assertIn("Would you like to order 30 plates instead?", state.get("final_response", ""))
        self.assertIsNone(state.get("order_id"))

    # ------------------------------------------------------------------------
    # 6. Customer declining confirmation
    # ------------------------------------------------------------------------
    def test_06_customer_declining_confirmation(self):
        thread_id = f"test_decline_{uuid.uuid4().hex[:6]}"
        
        # Turn 1: Valid proposal
        self._send_message("I want 2 plates of Paneer Tikka.", thread_id)
        
        # Turn 2: Customer declines
        state_2 = self._send_message("No, cancel it.", thread_id)
        self.assertEqual(state_2.get("order_status"), "rejected")
        self.assertIn("order has not been placed", state_2.get("final_response", ""))
        # Verify no order created in ORDERS_STORE
        self.assertEqual(len(ORDERS_STORE), 0)

    # ------------------------------------------------------------------------
    # 7. Order-creation failure
    # ------------------------------------------------------------------------
    def test_07_order_creation_failure(self):
        # Directly test take_order validation failure
        res = take_order(item_name="NonExistentItem", quantity=5)
        self.assertFalse(res["success"])
        self.assertEqual(res["error_code"], "UNKNOWN_ITEM")
        
        # Test out-of-stock item
        res_oos = take_order(item_name="Gulab Jamun", quantity=1)
        self.assertFalse(res_oos["success"])
        self.assertEqual(res_oos["error_code"], "INSUFFICIENT_INVENTORY")

    # ------------------------------------------------------------------------
    # 8. Cooking failure
    # ------------------------------------------------------------------------
    def test_08_cooking_failure_stops_serve(self):
        thread_id = f"test_cook_fail_{uuid.uuid4().hex[:6]}"
        
        # Turn 1: Propose order
        self._send_message("I want 2 plates of Garlic Naan.", thread_id)
        
        # Configure simulated cooking failure in graph state
        config = {"configurable": {"thread_id": thread_id}}
        self.agent.update_state(config, {"simulate_cook_failure": True})
        
        # Turn 2: Confirm order
        state_2 = self._send_message("Yes, confirm.", thread_id)
        self.assertEqual(state_2.get("order_status"), "accepted")
        self.assertEqual(state_2.get("cooking_status"), "failed")
        # Ensure serving was NOT executed
        self.assertNotEqual(state_2.get("serving_status"), "completed")
        self.assertIn("Kitchen error", state_2.get("final_response", ""))

    # ------------------------------------------------------------------------
    # 9. Serving failure
    # ------------------------------------------------------------------------
    def test_09_serving_failure(self):
        thread_id = f"test_serve_fail_{uuid.uuid4().hex[:6]}"
        
        # Turn 1: Propose order
        self._send_message("I want 2 plates of Garlic Naan.", thread_id)
        
        # Configure simulated serving failure
        config = {"configurable": {"thread_id": thread_id}}
        self.agent.update_state(config, {"simulate_serve_failure": True})
        
        # Turn 2: Confirm order
        state_2 = self._send_message("Yes, proceed.", thread_id)
        self.assertEqual(state_2.get("order_status"), "accepted")
        self.assertEqual(state_2.get("cooking_status"), "completed")
        self.assertEqual(state_2.get("serving_status"), "failed")
        self.assertIn("Fulfillment error", state_2.get("final_response", ""))

    # ------------------------------------------------------------------------
    # 10. Attempt to cook an order that was not accepted
    # ------------------------------------------------------------------------
    def test_10_cook_unaccepted_order(self):
        # Create an unaccepted order record manually
        fake_id = "ORD-UNACCEPTED"
        ORDERS_STORE[fake_id] = {
            "order_id": fake_id,
            "order_status": "pending",
            "cooking_status": "pending",
            "serving_status": "pending",
        }
        res = cook(fake_id)
        self.assertFalse(res["success"])
        self.assertEqual(res["error_code"], "INVALID_ORDER_STATUS")

        # Also test non-existent order
        res_none = cook("ORD-DOES-NOT-EXIST")
        self.assertFalse(res_none["success"])
        self.assertEqual(res_none["error_code"], "ORDER_NOT_FOUND")

    # ------------------------------------------------------------------------
    # 11. Attempt to serve an order that was not cooked
    # ------------------------------------------------------------------------
    def test_11_serve_uncooked_order(self):
        # Create an order that is accepted but cooking is still pending
        order_res = take_order("Garlic Naan", 2)
        self.assertTrue(order_res["success"])
        order_id = order_res["order_id"]
        
        # Directly attempt to serve before cooking
        serve_res = serve(order_id)
        self.assertFalse(serve_res["success"])
        self.assertEqual(serve_res["error_code"], "NOT_COOKED")

    # ------------------------------------------------------------------------
    # 12. Repeated request that could create a duplicate order
    # ------------------------------------------------------------------------
    def test_12_duplicate_order_protection(self):
        # Call take_order with explicit order_id twice
        order_id = "ORD-DEDUP-001"
        res1 = take_order("Garlic Naan", 2, order_id=order_id)
        self.assertTrue(res1["success"])
        initial_stock = MENU_INVENTORY["garlic naan"]["available_quantity"]

        # Call again with the same order_id (e.g. network retry)
        res2 = take_order("Garlic Naan", 2, order_id=order_id)
        self.assertTrue(res2["success"])
        self.assertEqual(res2["order_id"], order_id)
        # Verify inventory was NOT double-decremented
        self.assertEqual(MENU_INVENTORY["garlic naan"]["available_quantity"], initial_stock)

    # ------------------------------------------------------------------------
    # 13. Unrelated question
    # ------------------------------------------------------------------------
    def test_13_unrelated_question(self):
        thread_id = f"test_unrelated_{uuid.uuid4().hex[:6]}"
        state = self._send_message("What is the weather in Hyderabad?", thread_id)
        self.assertEqual(state.get("intent"), "unrelated")
        self.assertIn("restaurant ordering assistant", state.get("final_response", ""))
        self.assertIsNone(state.get("order_id"))
        self.assertEqual(len(ORDERS_STORE), 0)

    # ------------------------------------------------------------------------
    # 14. Menu or availability question without placing order
    # ------------------------------------------------------------------------
    def test_14_menu_or_availability_inquiry(self):
        thread_id = f"test_inquiry_{uuid.uuid4().hex[:6]}"
        state = self._send_message("What is the price of Paneer Tikka?", thread_id)
        self.assertEqual(state.get("intent"), "inquiry")
        self.assertIn("Paneer Tikka is INR 220", state.get("final_response", ""))
        self.assertIsNone(state.get("order_id"))
        self.assertEqual(len(ORDERS_STORE), 0)

    # ------------------------------------------------------------------------
    # 15. Fabricated / untrusted customer price is ignored
    # ------------------------------------------------------------------------
    def test_15_fabricated_price_is_ignored(self):
        thread_id = f"test_untrusted_price_{uuid.uuid4().hex[:6]}"
        # User claims 10 plates of Chicken Butter Masala at 50 rupees each (authoritative is 320)
        state_1 = self._send_message(
            "I want 10 plates of Chicken Butter Masala at 50 rupees each.",
            thread_id,
        )
        self.assertTrue(state_1.get("awaiting_confirmation"))
        # Unit price must be 320.0, NOT 50.0
        self.assertEqual(state_1.get("unit_price"), 320.0)
        self.assertEqual(state_1.get("total_price"), 3200.0)
        self.assertIn("INR 320.00 per plate", state_1.get("final_response", ""))
        self.assertIn("INR 3200.00", state_1.get("final_response", ""))


if __name__ == "__main__":
    unittest.main()
