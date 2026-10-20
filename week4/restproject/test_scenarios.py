"""
Test script for the 3 test scenarios from prompt2.md:
- TC1: Unrelated question -> Partial order -> Reject partial -> Unavailable dish -> END due to order retries
- TC2: Available order -> Cook fails once -> Cook succeeds -> Serve fails once -> Cook retries succeeds -> Serve succeeds -> Overall SUCCESS
- TC3: Partial order -> Reject partial & order available -> Cook fails once -> Cook succeeds -> Serve fails once -> Cook retries succeeds -> Serve fails again -> Cook retry exhausted -> Overall FAIL
"""

import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from restproject.agent import simulate_order, DEFAULT_MENU


def run_tc1():
    print("\n" + "=" * 70)
    print("▶ RUNNING TEST CASE 1 (TC1):")
    print("Scenario: Unrelated question -> Partial order -> Rejects partial -> Unavailable dish -> END due to order retry")
    print("=" * 70)

    inputs = [
        "What is quantum computing?",
        "I want 6 momos",
        "No, I don't want the partial order, I want to order another dish",
        "Give me sushi",
    ]

    state = simulate_order(inputs)

    print("\n--- TC1 Execution Dialogue ---")
    for msg in state["messages"]:
        sender = "Customer" if getattr(msg, "type", "") == "human" else "AI Agent"
        print(f"[{sender}]: {msg.content}")

    print("\n--- TC1 Verification ---")
    print(f"Final Status: {state.get('status')}")
    print(f"Final Result: {state.get('final_result')}")
    print(f"Order Retries Remaining: {state.get('order_retry_attempts')}")
    print(f"Available Quantity: {state.get('available_quantity')}")

    passed = (
        state.get("order_retry_attempts") == 0
        and "Order retry limit reached" in state.get("final_result", "")
    )
    print(f"TC1 PASSED: {passed}")
    return passed


def run_tc2():
    print("\n" + "=" * 70)
    print("▶ RUNNING TEST CASE 2 (TC2):")
    print("Scenario: Available order -> Cook fails once -> Cook succeeds -> Serve fails once -> Cook retries succeeds -> Serve succeeds -> Overall SUCCESS")
    print("=" * 70)

    inputs = [
        "I want 2 pizzas",
    ]

    # Cook: [False (initial fail), True (retry success), True (recook after serve fail)]
    # Serve: [False (initial serve fail), True (serve after recook success)]
    mock_cook = [False, True, True]
    mock_serve = [False, True]

    state = simulate_order(
        inputs,
        mock_cook_sequence=mock_cook,
        mock_serve_sequence=mock_serve,
    )

    print("\n--- TC2 Execution Dialogue ---")
    for msg in state["messages"]:
        sender = "Customer" if getattr(msg, "type", "") == "human" else "AI Agent"
        print(f"[{sender}]: {msg.content}")

    print("\n--- TC2 Verification ---")
    print(f"Final Status: {state.get('status')}")
    print(f"Final Result: {state.get('final_result')}")
    print(f"Cook Retries Remaining: {state.get('cook_retry_attempts')}")
    print(f"Serve Retries Remaining: {state.get('serve_retry_attempts')}")

    passed = (
        state.get("status") == "completed"
        and state.get("final_result") == "Order Completed Successfully"
    )
    print(f"TC2 PASSED: {passed}")
    return passed


def run_tc3():
    print("\n" + "=" * 70)
    print("▶ RUNNING TEST CASE 3 (TC3):")
    print("Scenario: Partial order -> Does not want partial -> Orders available -> Cook fails once -> Cook succeeds -> Serve fails -> Cook retries succeeds -> Serve fails again -> Cook retry exhausted -> Overall FAIL")
    print("=" * 70)

    inputs = [
        "I want 6 momos",
        "No, I don't want partial. Give me 2 pizzas instead",
    ]

    # Cook: [False (initial cook fail), True (cook retry success), True (recook after 1st serve fail)]
    # Serve: [False (1st serve fail), False (2nd serve fail)]
    mock_cook = [False, True, True]
    mock_serve = [False, False]

    state = simulate_order(
        inputs,
        mock_cook_sequence=mock_cook,
        mock_serve_sequence=mock_serve,
    )

    print("\n--- TC3 Execution Dialogue ---")
    for msg in state["messages"]:
        sender = "Customer" if getattr(msg, "type", "") == "human" else "AI Agent"
        print(f"[{sender}]: {msg.content}")

    print("\n--- TC3 Verification ---")
    print(f"Final Status: {state.get('status')}")
    print(f"Final Result: {state.get('final_result')}")
    print(f"Cook Retries Remaining: {state.get('cook_retry_attempts')}")
    print(f"Serve Retries Remaining: {state.get('serve_retry_attempts')}")

    passed = (
        state.get("status") == "failed"
        and ("Cook retries exhausted" in state.get("final_result", "") or "Serving failed" in state.get("final_result", ""))
        and state.get("cook_retry_attempts") == 0
    )
    print(f"TC3 PASSED: {passed}")
    return passed


if __name__ == "__main__":
    t1 = run_tc1()
    t2 = run_tc2()
    t3 = run_tc3()

    print("\n" + "=" * 70)
    print("🎯 FINAL RESULTS:")
    print(f"TC1: {'PASSED ✅' if t1 else 'FAILED ❌'}")
    print(f"TC2: {'PASSED ✅' if t2 else 'FAILED ❌'}")
    print(f"TC3: {'PASSED ✅' if t3 else 'FAILED ❌'}")
    print("=" * 70)
