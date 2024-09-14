import json
import re

class EntityScores:
    def __init__(self, name):
        self.name = name
        self.scores = []

    def add_score(self, score):
        self.scores.append(score)

    def __str__(self):
        return f"Entity: {self.name}\nScores:\n" + "\n".join(self.scores)

class EntityScoreCollection:
    def __init__(self):
        self.entities = {}

    def add_entity(self, entity_name):
        if entity_name not in self.entities:
            self.entities[entity_name] = EntityScores(entity_name)

    def add_score_to_entity(self, entity_name, score):
        if entity_name in self.entities:
            self.entities[entity_name].add_score(score)
        else:
            raise ValueError("Entity not found in the collection")
        
    def remove_entity(self, entity_name):
        if entity_name in self.entities:
            del self.entities[entity_name]
        else:
            raise ValueError("Entity not found in the collection")
        
    def find_entity(self, entity_name):
        if entity_name in self.entities:
            return str(self.entities[entity_name])
        else:
            raise ValueError("Entity not found in the collection")
    

    def __str__(self):
        return "\n".join(str(self.entities[name]) for name in sorted(self.entities))

    def to_json(self):
        return json.dumps({name: entity.scores for name, entity in self.entities.items()}, indent=4)


def clean_list(entities):
    """
    Removes empty elements and a specific element from the list.
    
    :param entities: list, the input list containing entity names and possibly empty elements.
    :return: list, the cleaned list without empty elements and the specified key entity.
    """
    # 过滤掉空字符串和特定的元素 '**Key Entities**:'
    cleaned_list = [item.strip() for item in entities if item.strip() and item != '**Key Entities**:' and item != '**Entity Event Hints**:' and '**start**' not in item and '**ending**' not in item]  
    return cleaned_list



def extract_step_one_content(front_pattern, back_pattern, text):
    """
    Extracts the section from '#### Step 1' to the first double newline using string methods.
    
    :param text: str, the input text from which to extract content.
    :return: str, the extracted content or an empty string if not found.
    """
    # 找到 "#### Step 1" 的起始位置
    start_index = text.find(front_pattern)
    if start_index == -1:
        return "Section 'front_pattern' not found"

    # 从找到的起始位置开始，查找后续的第一个双换行符 "\n\n"
    end_index = text.find(back_pattern, start_index)
    if end_index == -1:
        return "Section does not end properly with double newlines"
    if back_pattern == "\n\n":
        end_index += 2
    elif back_pattern == "**ending**":
        end_index +=10
    
    # 提取并返回这部分内容
    return text[start_index:end_index]



def extract_score(text):
    # 通过'--Score:'分割字符串
    parts = text.split('--Score:')
    # 如果正确分割，则第二部分包含分数
    if len(parts) > 1:
        score = parts[1].strip()  # 去除可能的前后空格
        return score
    return None  # 如果没有找到'--Score:'，返回None



def extract_entities_and_scores(text):


    # Extract sections for Key Entities and Event Hints
    front_pattern = "**start**"
    back_pattern = "**ending**"
    scores = extract_step_one_content(front_pattern, back_pattern, text)

    # Prepare to collect data
    entity_scores = EntityScoreCollection()
    entity_list = []

    # Process Entity Event Hints
    if scores:
        score_list = scores.split("\n")
        score_list = clean_list(score_list)
        entity = None  # 当前处理的实体
        for line in score_list:
            stripped_line = line.strip()
            if stripped_line.endswith(':'):
                if stripped_line[:-1] not in entity_list:
                    entity_scores.add_entity(stripped_line[:-1])
                entity = stripped_line[:-1]
            elif not re.match(r'\s*\d+\.', stripped_line):
                if stripped_line not in entity_list:
                    entity_scores.add_entity(stripped_line)
                entity = stripped_line

            elif entity is not None and re.match(r'\d+\.', stripped_line) and '--Score:' in stripped_line:
                entity_scores.add_score_to_entity(entity, extract_score(stripped_line))



    return entity_scores


def extract_entity_scores(text, entity_name):

    # Extract sections for Key Entities and Event Hints
    front_pattern = "**start**"
    back_pattern = "**ending**"
    scores = extract_step_one_content(front_pattern, back_pattern, text)

    entity_scores = None
    entity = None

    if scores:
        score_list = scores.split("\n")
        score_list = clean_list(score_list)
        for line in score_list:
            stripped_line = line.strip()
            if stripped_line.endswith(':'):
                if entity_name == stripped_line[:-1]:
                    entity = stripped_line[:-1]
                    entity_scores = EntityScores(entity)
            elif not re.match(r'\s*\d+\.', stripped_line):
                if entity_name == stripped_line:
                    entity = stripped_line
                    entity_scores = EntityScores(entity)

            elif entity is not None and re.match(r'\d+\.', stripped_line) and '--Score:' in stripped_line:
                entity_scores.add_score(extract_score(stripped_line))
    
    return entity_scores


def llama_extract_scores(text, hint_list):
    first_pattern = "### Response"
    front_pattern = "**start**"
    back_pattern = "**ending**"

    _, scores = text.split(first_pattern, 1)
    str_scores = extract_step_one_content(front_pattern, back_pattern, scores)
    lines = str_scores.split('\n')
    lines = clean_list(lines)

    entity = None
    entity_list = []
    entity_scores = {}

    print(lines)
    flag = True
    sum_score = 0 
    for line in lines:
        if not re.match(r'\d+\.', line):
            if entity is not None:
                if entity not in hint_list:
                    flag = False
                    break
                if len(hint_list[entity]) > 0:
                    aver = sum_score / len(hint_list[entity])
                else:
                    aver = 0
                entity_scores[entity] = aver
                sum_score = 0
            entity = line.strip()
            entity_list.append(entity)
            print(entity)
            if entity not in hint_list:
                flag = False
                break
        elif entity is not None and re.match(r'\d+\.', line):
            stripped_line = re.sub(r'^\d+\.\s*', '', line)
            parts = stripped_line.split('--', 1)
            if len(parts) == 2:
                hint, score = parts
                hint = hint.strip()
                if hint not in hint_list[entity]:
                    flag = False
                    break
                match = re.search(r'\d+\.\d+', score)
                if match:
                    extracted_score = float(match.group())
                else:
                    extracted_score = 0
            else:
                extracted_score = 0
            print(extracted_score)
            sum_score += extracted_score

            
    # if entity is not None and entity in hint_list:
    #     if len(hint_list[entity]) > 0: 
    #         aver = sum_score / len(hint_list[entity])
    #         entity_scores[entity] = aver
    #     else:
    #         aver = 0
    #         entity_scores[entity] = aver


    
    return flag, entity_scores


def read_file_and_process(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            entity_hints = extract_entities_and_scores(content)
            print(entity_hints)
            # 将结果输出为JSON
            json_output = entity_hints.to_json()
            print(json_output)
            return json_output
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")




def average_entities_scores(entity_scores):
    all_total_aver = []
    entity_list = []
    for entity in entity_scores.entities.values():
        sum_score = 0
        entity_list.append(entity.name)
        print(f"Entity: {entity.name}")
        for score in entity.scores:
            print(f"Score: {score}")
            sum_score += float(score)
        if len(entity.scores) != 0:
            aver = sum_score /len(entity.scores)
        else:
            aver = 0
        all_total_aver.append(aver)
    entityandscores = dict(zip(entity_list, all_total_aver))
    return entityandscores, entity_list


def average_entity_scores(entity_scores):
    sum_score = 0
    if entity_scores is None:
        print("Error: entity_scores is None")
        return 0  # 或者根据你的程序需要返回其他合适的值
    for score in entity_scores.scores:
        sum_score += float(score)
    if len(entity_scores.scores) != 0:
        aver = sum_score / len(entity_scores.scores)
    else:
        aver = 0
    return aver



if __name__ == "__main__":
    # text = '\n    ### Instruction:\n    Suppose you are a seasoned mathematician tasked with analyzing a mathematical problem. First, identify key entities relevant to solving the problem and then generate multiple event hints for each entity to uncover both explicit and implicit knowledge about the mathematical issue. This will help in forming a detailed understanding of the problem. Please ensure that your event hints are logically connected to the entities and are helpful in solving the mathematical problem.\n\n    ----\n    ### Input:\n    "Problem": "There are 87 oranges and 290 bananas in Philip\'s collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups. How big is each group of bananas?"\n    We aim to score the event hints of key entities to fully reveal their priority relevance to the math problem.\n\n\n    ### Hints Score Response:\n    "Hints Score":\n    **start**\n        \n        Number of groups for bananas:\n        \n        1. The bananas are to be divided into 2 groups. --Score:"0.9"---\n        \n        \n        Total number of bananas:\n        \n        1. The problem states that there are 290 bananas in total. --Score:"1.0"---\n        \n        \n    **ending**\n    \n    ### Input:\n    "Problem": "Marco and his dad went strawberry picking. Marco\'s dad\'s strawberries weighed 11 pounds. If together their strawberries weighed 30 pounds. How much did Marco\'s strawberries weigh?"\n    We aim to score the event hints of key entities to fully reveal their priority relevance to the math problem.\n\n\n    ### Hints Score Response:\n    "Hints Score":\n    **start**\n        \n        Total weight of Marco\'s dad\'s strawberries:\n        \n        1. The problem states that Marco\'s dad\'s strawberries weighed 11 pounds. --Score:"1.0"---\n        \n        \n        Total weight of Marco\'s strawberries:\n        \n        1. Let\'s denote the weight of Marco\'s strawberries as x. --Score:"0.9"---\n        \n        \n        Total weight of both their strawberries:\n        \n        1. The combined weight of Marco\'s and his dad\'s strawberries is 30 pounds. --Score:"1.0"---\n        \n        \n    **ending**\n    \n    ### Input:\n    "Problem": "Edward spent $ 6 to buy 2 books each book costing him the same amount of money. Now he has $ 12. How much did each book cost?"\n    We aim to score the event hints of key entities to fully reveal their priority relevance to the math problem.\n\n\n    ### Hints Score Response:\n    "Hints Score":\n    **start**\n        \n        Number of books purchased:\n        \n        1. Edward bought 2 books. --Score:"1.0"---\n        \n        \n        Total amount Edward has now:\n        \n        1. Edward now has $12. --Score:"0.9"---\n        \n        \n        Total amount spent by Edward:\n        \n        1. Edward spent $6 to buy the books. --Score:"0.8"---\n        \n        \n    **ending**\n    \n    ### Input:\n    "Problem": "Jerry had 7 books and 3 action figures on a shelf in his room. Later he added 2 more action figures to the shelf. How many more books than action figures were on his shelf?"\n    For this mathematical problem, Just score the event hints, no summary or synthesis is needed. Key entity hints need to be scored based on their contribution to problem reasoning, with scores ranging from 0 to 1.\n\n    ### Response:\n    "Hints Score":\n    **start**\n        \n        Number of books in Jerry\'s collection\n        \n        1. Jerry had 7 books in his collection. --Score:"0.5"\n        \n        \n        Number of action figures in Jerry\'s collection\n        \n        1. Jerry had 3 action figures in his collection. --Score:"0.5"\n        \n        \n    **ending**"\n    '
    # hint_list = {
    #     "Number of books in Jerry's collection": ['Jerry had 7 books in his collection.'], 
    #     "Number of action figures in Jerry's collection": ['Jerry had 3 action figures in his collection.']}
    # # file_path = './test.txt'
    # # read_file_and_process(file_path)
    # print(llama_extract_scores(text, hint_list))

    hint_list = {
        'Number of packs of dvds you can buy': [
            'Given the amount of money available, you can buy 4 packs of dvds.',
            'The problem states that you have 104 dollars.'
        ]
    }

    # 定义空的 final_hints 列表
    final_hints = []

    # 提取 hint_list 中的第一个元素的值并作为第一个元素添加到 final_hints
    # 我们使用 .values() 方法获取字典的值，然后用 next(iter(...)) 来取得第一个值
    first_hint = next(iter(hint_list.values()))

    # 将获取到的值（一个列表）作为一个整体元素添加到 final_hints 中
    final_hints.append(first_hint)

    # 输出 final_hints 以验证
    print(final_hints)

