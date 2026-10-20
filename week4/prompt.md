# PROJECT: RESTAURANT AI AGENT

## 1. OBJECTIVE

Build a restaurant ordering AI agent using Python, Groq, and LangGraph.

The agent should accept natural-language food orders, validate them against the restaurant's menu and inventory, process valid orders, manage cooking and serving operations, and communicate the final result to the customer.

The goal is to automate restaurant order processing while reducing manual work.

This is initially a learning project. Use simulated menu data, inventory, cooking, and serving operations. Keep the architecture modular so a real database can be integrated later.

## 2. TECHNOLOGY STACK

* Programming language: Python
* LLM provider: Groq
* LLM model: openai/gpt-oss-120b
* Agent framework: LangGraph
* Environment configuration: python-dotenv
* Load environment variables using `load_dotenv()`
* Store API credentials in a `.env` file.

Do not hardcode API keys or expose secrets in logs or responses.

Use the Groq Python client and LangGraph's supported APIs. Verify the installed package interfaces rather than assuming outdated syntax.

## 3. RESTAURANT MENU AND INVENTORY

Create a sample menu containing:

* Item name
* Price per plate
* Available quantity
* Availability status

For the initial implementation, menu and inventory data can be stored in Python dictionaries.

The menu and inventory are the application's source of truth.

The LLM must never invent menu items, prices, or available quantities.

The customer should provide the food item and quantity, not the price. The application must retrieve the authoritative price from the menu.

Calculate the total price using:

`total_price = quantity * unit_price`

Validate all quantities as positive integers.

Reject unknown menu items and insufficient inventory without creating an invalid order.

## 4. TOOLS

Implement the following three tools.

### TOOL 1: take_order

**Purpose:** Validate and record a customer's food order.

**Inputs:**

* `item_name`
* `quantity`

**Responsibilities:**

1. Check whether the requested item exists in the menu.
2. Validate that the quantity is a positive integer.
3. Check whether the requested quantity is available.
4. Retrieve the item's price from the menu.
5. Calculate the total price.
6. Create an order record when validation succeeds.
7. Return a structured result describing the outcome.

**Expected successful output:**

* Success status
* Order ID
* Item name
* Confirmed quantity
* Unit price
* Total price
* Order status

**Expected failure output:**

* Failure status
* Error code
* Human-readable error message
* Available quantity, when relevant

The tool must not accept a customer-supplied price as authoritative.

Do not create an order if validation fails.

Avoid duplicate order creation if an operation is retried.

### TOOL 2: cook

**Purpose:** Simulate cooking an accepted order.

**Input:**

* `order_id`

**Responsibilities:**

1. Verify that the order exists.
2. Verify that the order has been accepted.
3. Verify that the order has not already been cooked.
4. Simulate the cooking operation.
5. Update the cooking status only when the operation succeeds.
6. Return a structured result to the graph.

Possible cooking statuses:

* `pending`
* `in_progress`
* `completed`
* `failed`

For this learning project, cooking can be simulated without connecting to physical restaurant equipment.

The tool must not create orders or modify confirmed prices and quantities.

### TOOL 3: serve

**Purpose:** Simulate serving or fulfilling a successfully cooked order.

**Input:**

* `order_id`

**Responsibilities:**

1. Verify that the order exists.
2. Verify that cooking has completed successfully.
3. Verify that the order has not already been served.
4. Simulate the serving operation.
5. Update the fulfillment status only when the operation succeeds.
6. Return a structured result to the graph.

Possible serving statuses:

* `pending`
* `completed`
* `failed`

The tool must reject attempts to serve orders that have not been cooked successfully.

Do not implement table numbers, table validation, or table-related interactions. Serving should require only the order ID.

## 5. AGENT BEHAVIOR

The agent should understand the user's request and determine whether it is related to restaurant ordering.

It should extract the food item and quantity when available.

### Scenario A: Valid order

User:
"I want 10 plates of Chicken Butter Masala."

Expected workflow:

1. Extract the item and quantity.
2. Validate the order against the menu and inventory.
3. If valid, present the item, quantity, unit price, and total price to the customer.
4. Ask the customer to confirm the order before creating it.
5. After confirmation, execute `take_order`.
6. If order creation succeeds, execute `cook`.
7. If cooking succeeds, execute `serve`.
8. Return a final confirmation containing the order ID, item, quantity, total price, and final status.

If the customer has not confirmed the order, do not create it or begin cooking.

### Scenario B: Missing quantity

User:
"I want Chicken Biryani."

Expected response:
"How many plates of Chicken Biryani would you like?"

Preserve the identified item in the graph state so the user does not need to repeat it.

Do not guess the quantity.

### Scenario C: Unknown menu item

User:
"I want horse meat."

Expected response:
"Sorry, that item isn't on our menu. Here are our available food options: ..."

Show the actual menu.

Do not call `cook` or `serve`.

### Scenario D: Insufficient inventory

User:
"I want 50 plates of Chicken Biryani."

Suppose only 30 plates are available.

Expected response:
"We currently have only 30 plates of Chicken Biryani available. Would you like to order 30 plates instead?"

Do not silently change the requested quantity.

Wait for the customer's response before proceeding.

### Scenario E: Invalid quantity

User:
"I want zero plates of Chicken Biryani."

Expected response:
"Please provide a quantity greater than zero."

Do not create an order.

### Scenario F: Unrelated question

User:
"What is the weather in Hyderabad?"

Expected response:
"I'm a restaurant ordering assistant. I can help you with our menu, prices, availability, and food orders."

Do not invoke order-processing tools.

However, legitimate restaurant questions about menu items, prices, and availability should be answered without requiring the customer to place an order.

### Scenario G: Customer declines confirmation

If the customer declines the proposed order, do not create the order or begin cooking.

Respond politely and end the current ordering attempt.

### Scenario H: Tool failure

If any tool fails:

* Do not claim the operation succeeded.
* Communicate the failure clearly.
* Preserve the existing order state.
* Do not blindly repeat operations that could create duplicate orders or duplicate fulfillment actions.

## 6. LANGGRAPH ARCHITECTURE

Use LangGraph to control the order-processing workflow.

Before implementing the code, propose a Mermaid diagram showing the graph.

The intended logical workflow is:

START
|
v
Understand Customer Request (LLM)
|
v
Validate Request and Required Information
|
+---- Invalid or Missing Information
|              |
|              v
|        Ask Customer
|              |
|              v
|             END
|
+---- Valid Order Details
|
v
Present Order Summary
|
v
Wait for Confirmation
|
+------+------+
|             |
Declined      Confirmed
|             |
v             v
END        take_order
|
v
Check Order Result
|
+------+------+
|             |
Failed        Success
|             |
v             v
END           cook
|
v
Check Cook Result
|
+------+------+
|             |
Failed        Success
|             |
v             v
END           serve
|
v
Check Serve Result
|
v
Final Response
|
v
END

Treat this diagram as the intended behavior, not as a requirement to implement every box as a separate node.

Use the simplest reasonable graph structure.

### Graph requirements

* Use a shared graph state.
* Use conditional edges for validation results and operation outcomes.
* Enforce the order `take_order → cook → serve`.
* Do not let the LLM independently skip required operations.
* Do not call `cook` if order creation fails.
* Do not call `serve` if cooking fails.
* Do not call `take_order` until the customer has confirmed the order.
* Route missing information back to a customer-interaction step.
* End the workflow when the request is unrelated, the customer declines, or an unrecoverable error occurs.

Use LLM reasoning where natural-language interpretation is needed and deterministic graph transitions where reliable business-process control is required.

## 7. STATE MANAGEMENT

Define an appropriate LangGraph state schema.

It should include fields such as:

* `messages`
* `item_name`
* `quantity`
* `unit_price`
* `total_price`
* `order_id`
* `order_status`
* `cooking_status`
* `serving_status`
* `customer_confirmation`
* `available_quantity`
* `error`

Use appropriate types and sensible defaults.

The state should preserve information between graph nodes and customer follow-up interactions.

Use explicit status fields rather than relying on natural-language tool responses to determine whether an operation succeeded.

If using LangGraph persistence to pause for customer confirmation, configure an appropriate checkpointer and stable thread identifier. Ensure the application can resume the correct conversation.

Do not assume that ordinary in-memory state persists across application restarts.

## 8. SYSTEM PROMPT

Create a system prompt for the restaurant ordering assistant.

It should define:

* The agent's role and scope.
* How to interpret food-order requests.
* How to identify missing information.
* When to display the menu.
* How to handle unavailable items and quantities.
* How to present an order summary.
* How to obtain customer confirmation.
* How to communicate tool failures.
* How to provide accurate final order summaries.

The system prompt should guide the LLM's behavior, but it must not be the only mechanism enforcing business rules.

Validate prices, inventory, customer confirmation, order status, and operation sequencing in application code.

## 9. RELIABILITY AND SAFETY

Implement:

* Input validation.
* Structured tool arguments and results.
* Appropriate exception handling.
* Bounded retries.
* Protection against duplicate order creation.
* Explicit order-status transitions.
* Useful error messages.
* Logging of important workflow transitions without exposing secrets.

For the initial learning version, simulate cooking and serving using deterministic Python functions.

Do not implement unrestricted code execution or use arbitrary `eval()`.

Keep the code modular and readable.

## 10. TESTING

Test at least the following cases:

1. A valid order with customer confirmation.
2. An order with a missing quantity.
3. An unknown menu item.
4. A zero or negative quantity.
5. A quantity exceeding inventory.
6. A customer declining confirmation.
7. An order-creation failure.
8. A cooking failure.
9. A serving failure.
10. An attempt to cook an order that was not accepted.
11. An attempt to serve an order that was not cooked.
12. A repeated request that could create a duplicate order.
13. An unrelated question.
14. A menu or availability question that does not create an order.
15. A request containing an item and quantity but an untrusted or fabricated price.

Verify that failed operations do not cause later workflow steps to execute.

## 11. IMPLEMENTATION PROCESS

Work in the following stages.

Stage 1: Review the requirements and identify ambiguities.

Stage 2: Propose the LangGraph architecture and Mermaid diagram.

Stage 3: Define the state schema and menu structure.

Stage 4: Define the tools, input schemas, output schemas, and validation rules.

Stage 5: Design the system prompt.

Stage 6: Implement the graph and its conditional routing.

Stage 7: Implement tests for success paths and failure paths.

Stage 8: Provide execution instructions and explain the architecture.

Do not generate the entire project immediately.

First, present the architecture, graph diagram, state schema, and tool contracts. Explain why the proposed design satisfies the requirements.

Wait for architectural review before proceeding to implementation.

## DELIVERABLES

Provide:

* Python source code.
* LangGraph architecture diagram.
* System prompt.
* Tool definitions.
* State schema.
* Tests.
* `.env.example` containing placeholder values only.
* `requirements.txt`.
* README with setup and execution instructions.
* An explanation of the nodes, edges, conditional routing, and tool interactions.

Prioritize correctness, simplicity, readability, and learning value over unnecessary complexity.
