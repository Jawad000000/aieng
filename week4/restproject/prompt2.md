iam building a resturant order management ai system using langgraph.

i will explain you very clearly about the architecture I will also explain you the notes and the edges
In the end I will give you test cases to simulate and tell me whether they will pass or not

the rough plan:
There will be a user when the code runs the user will give an order This will be taken as an input which uh the user or every type
The order as of now should contain a dish dish and a required quantity
For simplicity purpose for now we will limit the order to one particular dish and whatever on the user.
The LLM will have to extract the order and the quantity from the user's input.
If the user's input is unrelated to the food  ordering the LLM should not proceed until the user that it is a restaurant ordering AI agent not an general purpose LLM

Was the LLM has the order and dish quantity
It will send that to a node called order confirm.
The tasks of this order confirm is to look at the menu {I will tell you the menu later in this prompt. }
This node will decide one of the three cases
 the order is available. 
 The order is partially available means quantity is not sufficient.
  The order is not available at all dish being not in the menu or zero quantity. 
  It will put this in the status of the state {I will give you the exact content of Langgraph state as well }
  once the LLM receive this if the standard is confirmed {full available }should call another node called cook.
  if the status is partial or not available it should again prompt the user to confirm the order and should loop until the order is confirmed or user cancels the order.
  the user can place either new order or can confirm if he wants to go ahead with the partial order.

  this order retires will be limited to 3 attemps meaning if After three attempts the user is not satisfied then the system will come to an end node

now when The cook node is called there can be two cases.
Either the cook is done, then the status will be ready.
And if the coop fails, we can use {probablity function}- give 40% chances of fail and 60% chacnes of successs

if the cook fails tehre should be 1 more attempt allowed for cook to succed.

If the cook fails even after this then the LLM should issue an apology to the user and come to end state. If the cook succeeds, the status will be ready and the next node will be called which is serve
 similar to the cook this also has two cases:
 serve pass and serve fail

 This also has to retry attempt

 if the serve fails two time then the LLM should issue an apology to the user and come to the end state.
 If the serve succeeds the status should become completed and the LLM should issue a message to the user saying that your order is completed. 
..
if the serve fails for the first time or has not exhausted it retries at him then it should call the cop so that the llm will retry cooking. 
Note if the cook has exhausted its retry at him then it should not cook again and the LLM should issue an apology and come to an end state
..
if the serve fails and cook has already exhausted its retry attemp then the LLM should issue an apology and come to end state.


Now coming to Langgraph
there should be a annotated message it should be llm and user

there should be order details
dish name as str
quantity as int
availabe quantity as string
order_confirm will write the available quantity by reading the menu
then the llm should get to know the order confirm status by reading the state
if the dish is not available in the menu the order confim should write 0 as available quantity
tehn tehre should be status

each node will update the status as specified in the above rules. 
then order we try attempts which are three 
Then order we try attempts which are three
cook retry attemps which are 2

 Serve retry attempts which are 2
 Each time a failure happens and a node is retrying it should decrement the counter 

 If any retry counter become zero then it means it is over. 
 LLM should understand whether it has to give a retry or issue apology by reading this counter next slide

 in the end there final result whether the order was complete or not

 write the code and ask if there are any open questions from your side then ask
 this is menu
 burger 10
 pizza 20
 momos 4
 use groq as llm too
 i will give you some test scenarious to test later on
 do only what was ask in the prompt.

TC1 a user aska unrealted question then he palces a order which is partial then he rejects the partial and wants to order again again he ordes a dish which is not avilabl expected. END due to order retry

TC2 user places a order which can be availeble cook makes it fails once cooks again cucess serve fails once cook retries success serve does success overall success

TC3 user orders partal doesn not want partial orders again fuylly available cooks kakes it fails makes again success serve fails cook retries success serve fails again cook does not retry ue to cook retru exhaist overall FAIl