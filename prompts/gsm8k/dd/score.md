<system>
You will be presented with a mathematical problem labeled as "gsm8k". In understanding this math problem, key entities within the math question and corresponding event hints for these entities are provided to help you truly understand the problem. For each entity event hints, assess its contribution for problem reasoning to this math problem. This forms a sequence of [Problem] -> [Entity Event Hints] -> [Scoring].

### Goal
Your primary goal is to score the event hints of key entities to fully reveal their priority relevance to the math problem.

### Key Priorities
1. **Score Format**: When outputting scores, ensure that only the entities, event hints, and scores are output. Remember, you must follow the format below without adding any words before the start or after the end:
**Entity and Event Scoring**:
  **start**
  [Event Hint --Score:]
  **ending**

</system>

## Problem: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? 

### Entity Event Hints:
Janet: 
1. She consumes 3 eggs for breakfast. 
2. She bakes muffins with 4 eggs daily 
3. Janet sells the remaining eggs at the farmers' market. 
4. She goes to the farmers' market every day.
Ducks:
1. Ducks lay 16 eggs per day.
2. Ducks are the source of eggs for Janet.
3. Janet's ducks produce eggs regularly.

Just score the event hints.

#### Score:Just score the event hints, no summary or synthesis is needed. Key entity hints need to be scored based on their contribution to problem reasoning, with scores ranging from 0 to 1.
- **Entity and Event Scoring**:
  **start**
  Janet:
  1. She eats 3 eggs for breakfast. --Score:0.7
  2. She bakes muffins with 4 eggs daily. --Score:0.6
  3. Janet sells the remaining eggs at the farmers' market --Score:0.8
  4. She goes to the farmers' market every day. --Score:0.9
  Ducks:
  1. Ducks lay 16 eggs per day. --Score:0.8
  2. Ducks are the source of eggs for Janet. --Score:0.6
  3. Janet's ducks produce eggs regularly. --Score:0.7
  **ending**