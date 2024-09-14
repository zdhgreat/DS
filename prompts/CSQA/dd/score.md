<system>
You will be presented with an open-domain question labeled as "CommonsenseQA". In understanding this question, key entities and corresponding event hints will be provided to help you truly grasp the problem. For each entity event hints, assess its contribution for problem reasoning to this math problem. This forms a sequence of [Problem] -> [Entity Event Hints] -> [Scoring].

### Goal
Your primary goal is to score the event hints of key entities to fully reveal their relevance to the priorities of the open-domain question.

### Key Priorities
1. **Score Format**: When outputting scores, ensure that only the entities, event hints, and scores are output. Remember, you must follow the format below without adding any words before the start or after the end:
**Entity and Event Scoring**:
  **start**
  [Event Hint --Score:]
  **ending**

</system>

## Problem: A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?

### Entity Event Hints:
Revolving Door:
1. Revolving doors are used in various public buildings to manage both the flow of people and energy efficiency by minimizing air exchange.
2. These doors can control the rate of entry and exit, making it easier to monitor and secure entrances in busy or sensitive areas.
Security Measure:
1. As a security measure, revolving doors prevent tailgating in sensitive buildings where controlling access is crucial, such as banks and corporate buildings.
2. Their design can integrate with security systems like cameras and guards, enhancing overall security protocols.
High-security Locations:
1. Banks often use revolving doors to enhance security, preventing quick exits and controlling access.
2. Department stores and malls use revolving doors for similar reasons, especially in urban areas where theft and security are significant concerns.
3. The presence of revolving doors in such locations can deter theft and unauthorized access, aligning with stringent security measures.

Just score the event hints.

#### Score:Just score the event hints, no summary or synthesis is needed. Key entity hints need to be scored based on their contribution to problem reasoning, with scores ranging from 0 to 1.
- **Entity and Event Scoring**:
  **start**
  Revolving Door:
  1. Revolving doors are used in various public buildings to manage both the flow of people and energy efficiency by minimizing air exchange. --Score: 0.7
  2. These doors can control the rate of entry and exit, making it easier to monitor and secure entrances in busy or sensitive areas. --Score: 0.9
  Security Measure:
  1. As a security measure, revolving doors prevent tailgating in sensitive buildings where controlling access is crucial, such as banks and corporate buildings. --Score: 1.0
  2. Their design can integrate with security systems like cameras and guards, enhancing overall security protocols. --Score: 0.9
  High-security Locations:
  1. Banks often use revolving doors to enhance security, preventing quick exits and controlling access. --Score: 1.0
  2. Department stores and malls use revolving doors for similar reasons, especially in urban areas where theft and security are significant concerns. --Score: 0.8
  3. The presence of revolving doors in such locations can deter theft and unauthorized access, aligning with stringent security measures. --Score: 0.8
  **ending**