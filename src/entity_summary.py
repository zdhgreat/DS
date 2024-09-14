import json
import re



def find_optimal(entites_hints, entities_scores):
    first_entity = None
    second_entity = None
    first_score = 1
    second_score = 1
    match = False
    if len(entites_hints.entities) == len(entities_scores):
        if len(entites_hints.entities) == 2:
            first_entity = list(entites_hints.entities.values())[0].name
            second_entity = list(entites_hints.entities.values())[1].name
        for entity_hint in entites_hints.entities:
            if entity_hint not in entities_scores:
                break
            if entities_scores[entity_hint] <= first_score:
                first_score = entities_scores[entity_hint]
                first_entity = entity_hint
        
        for entity_hint in entites_hints.entities:
            if entity_hint not in entities_scores:
                break
            if entities_scores[entity_hint] <= second_score and entity_hint != first_entity:
                second_score = entities_scores[entity_hint]
                second_entity = entity_hint   

        if first_entity == None or second_entity == None:
            first_entity = list(entites_hints.entities.values())[0].name
            second_entity = list(entites_hints.entities.values())[1].name

            
        match = True  
    else:
        print("no match! But you can review again!")
    

    return first_entity, second_entity, match 




def alter_find_optimal(entites_hints, entities_scores):
    first_entity = None
    second_entity = None
    first_score = 0
    second_score = 0
    match = False
    if len(entites_hints.entities) == len(entities_scores):
        if len(entites_hints.entities) == 2:
            first_entity = list(entites_hints.entities.values())[0].name
            second_entity = list(entites_hints.entities.values())[1].name
        for entity_hint in entites_hints.entities:
            if entity_hint not in entities_scores:
                break
            if entities_scores[entity_hint] >= first_score:
                first_score = entities_scores[entity_hint]
                first_entity = entity_hint
        
        for entity_hint in entites_hints.entities:
            if entity_hint not in entities_scores:
                break
            if entities_scores[entity_hint] >= second_score and entity_hint != first_entity:
                second_score = entities_scores[entity_hint]
                second_entity = entity_hint   

        if first_entity == None or second_entity == None:
            first_entity = list(entites_hints.entities.values())[0].name
            second_entity = list(entites_hints.entities.values())[1].name

            
        match = True  
    else:
        print("no match! But you can review again!")
    

    return first_entity, second_entity, match 





def llama_find_optimal(entities_hints, entities_scores):
    # Initialize variables to store the optimal entities
    first_entity = None
    second_entity = None
    match = False

    # Ensure the length of entities_hints matches the length of entities_scores
    if len(entities_hints) != len(entities_scores):
        return first_entity, second_entity, match
    
    # Sort entities based on scores in ascending order
    sorted_entities = sorted(entities_scores.items(), key=lambda x: x[1])

    # Iterate through sorted entities to find the two with the lowest scores
    for entity, score in sorted_entities:
        if entity in entities_hints:
            if first_entity is None:
                first_entity = entity
                first_score = score
            elif second_entity is None:
                second_entity = entity
                second_score = score
                match = True
                break



    return first_entity, second_entity, match
