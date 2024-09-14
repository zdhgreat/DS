import json
import re
from collections import OrderedDict

class EntityHints:
    def __init__(self, name):
        self.name = name
        self.hints = []

    def add_hint(self, hint):
        self.hints.append(hint)

    def __str__(self):
        numbered_hints = "\n".join(f"{idx + 1}. {hint}" for idx, hint in enumerate(self.hints))
        return f"{self.name}\n" + numbered_hints
    
    def only_hint(self):
        numbered_hints = "\n".join(f"{idx + 1}. {hint}" for idx, hint in enumerate(self.hints))
        return numbered_hints

class EntityHintCollection:
    def __init__(self):
        self.entities = {}

    def add_entity(self, entity_name):
        if entity_name not in self.entities:
            self.entities[entity_name] = EntityHints(entity_name)

    def add_hint_to_entity(self, entity_name, hint):
        if entity_name in self.entities:
            self.entities[entity_name].add_hint(hint)
        else:
            raise ValueError("Entity not found in the collection")
        
    def find_entity(self, entity_name):
        if entity_name in self.entities:
            return str(self.entities[entity_name])
        else:
            raise ValueError("Entity not found in the collection")
    
    def remove_entity(self, entity_name):
        if entity_name in self.entities:
            del self.entities[entity_name]
        else:
            raise ValueError("Entity not found in the collection")
    
    def output_hint(self):
        if len(self.entities.values()) == 1:
            return self.entities.only_hint()
        else:
            raise ValueError("Entity not found in the collection")
    
    def output_entities(self):
        return "\n".join(name for name in sorted(self.entities))

    def __str__(self):
        return "\n".join(str(self.entities[name]) for name in sorted(self.entities))

    def to_json(self):
        return json.dumps({name: entity.hints for name, entity in self.entities.items()}, indent=4)


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
    if front_pattern == "**start**":
        start_index += 9

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



def extract_entitiy_and_hints(entities_hints, text):
    front_pattern = "**start**"
    back_pattern = "**ending**"
    event_hints = extract_step_one_content(front_pattern, back_pattern, text)
    

    if event_hints:
        hints = event_hints.split("\n")
        hints = clean_list(hints)
        entity = None
        for hint in hints:
            line = hint.strip()
            if (line.endswith(':') or ':' in line) and entity == None:
                entities_hints.add_entity(line[:-1])
                entity = line[:-1]
            elif re.match(r'\d+\.', line):
                updated_line = re.sub(r'^\d+\.\s*', '', line)
                entities_hints.add_hint_to_entity(entity, updated_line)
    
    return entities_hints, entity


def remove_number_prefix(input_string):
    # 使用正则表达式匹配并替换可能的序号和冒号
    # 正则表达式解释：'^\d+\.\s*' 匹配开头的数字和点，后面可能跟着空格
    # '|^' 表示或者匹配字符串开头
    # '\s*:\s*' 匹配冒号前后的任意空白字符
    cleaned_string = re.sub(r'^\d+\.\s*|^', '', input_string)
    cleaned_string = re.sub(r'\s*:\s*', '', cleaned_string)
    return cleaned_string

def extract_entities_and_hints(text):


    # Extract sections for Key Entities and Event Hints
    front_pattern = "**start**"
    back_pattern = "**ending**"
    # front_pattern1 = "**start**"
    # front_pattern2 = "Step 2"
    # back_pattern1 = "\n\n"
    # back_pattern2 = "**ending**"
    part1, part2 = text.split('Entity Event Hints', 1)
    key_entities = extract_step_one_content(front_pattern, back_pattern, part1)
    event_hints = extract_step_one_content(front_pattern, back_pattern, part2)
    #key_entities = re.findall(pattern1, text, re.DOTALL)
    #entity_hints = re.findall(pattern2, text, re.DOTALL)
    #print(key_entities)
    #print(entity_hints)

    
    entity_hints = EntityHintCollection()
    # Prepare to collect data
    entity_list = []

    # Process Key Entities
    if key_entities:
        entity_list = [entity.strip('- ').strip() for entity in key_entities.split("\n") if entity.strip()]
        entity_list = clean_list(entity_list)
        #entity_list = [entity.strip() for entity in entities.split('\n') if entity.strip()]

    for entity in entity_list:
        entity_hints.add_entity(entity)

    # Process Entity Event Hints
    if event_hints:
        hint = event_hints.split("\n")
        hint = clean_list(hint)
        entity = None  # 当前处理的实体
        for line in hint:
            stripped_line = line.strip()
            if stripped_line.endswith(':'):
                clean_string = remove_number_prefix(stripped_line)
                if clean_string not in entity_list:
                    entity_hints.add_entity(clean_string)

                entity = clean_string

            elif entity is not None and re.match(r'\d+\.', stripped_line):
                updated_stripped_line = re.sub(r'^\d+\.\s*', '', stripped_line)
                entity_hints.add_hint_to_entity(entity, updated_stripped_line)



            #int_list = [hint.strip('- ').strip() for hint in hints.split(":\\n") if hint.strip()]
            #hint_list = clean_list(hint_list)
        # entity_blocks = re.split(r'\n\s*\n', entity_hints[0].strip())
        # for block in entity_blocks:
        #     entity_name = re.findall(r'(.*?):\n', block)
        #     events = re.findall(r'\d+\.\s*(.*?)\n', block)
        #     if entity_name:
        #         event_hints[entity_name[0].strip()] = events
    
    to_remove = []
    for en in entity_hints.entities:
        length = len(entity_hints.entities[en].hints)
        if length == 0:
            to_remove.append(en)

    for en in to_remove:
        entity_hints.remove_entity(en)



    return entity_hints

def read_file_and_process(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            entity_hints = extract_entities_and_hints(content)
            print(entity_hints)
            # 将结果输出为JSON
            json_output = entity_hints.to_json()
            print(json_output)
            return json_output
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def llama_extract_entites(entitis):
    entity_list = re.findall(r'- (.+)', entitis)
    return entity_list


def llama_extract_hints(text):
    first_pattern = "### Response"
    front_pattern = "**start**"
    back_pattern = "**ending**"
    
    entity_hints = {}
    
    _, hints = text.split(first_pattern, 1)
    str_hints = extract_step_one_content(front_pattern, back_pattern, hints)
    lines = str_hints.split('\n')
    lines = clean_list(lines)
    
    entity = None
    for line in lines:
        if line.endswith(":\""):
            stripped_line, _ = line.split(":\"", 1)
            entity = stripped_line
            entity_hints[entity] = []
        elif entity is not None and re.match(r'\d+\.', line):
            stripped_line = re.sub(r'^\d+\.\s*', '', line)
            entity_hints[entity].append(stripped_line)
    
    return entity_hints


def llama_extract_hint(text):
    lines = text.split('\n')
    lines = clean_list(lines)
    entity_hints = {}
    entity = None
    first_entity = None
    for line in lines:
        if line.endswith(":"):
            stripped_line, _ = line.split(":", 1)
            entity = stripped_line
            entity_hints[entity] = []
            if len(entity_hints) == 1:
                first_entity = entity
        elif not re.match(r'\d+\.', line):
            stripped_line = line
            entity = stripped_line
            entity_hints[entity] = []
            if len(entity_hints) == 1:
                first_entity = entity
        elif entity is not None and re.match(r'\d+\.', line):
            stripped_line = re.sub(r'^\d+\.\s*', '', line)
            entity_hints[entity].append(stripped_line)

    entity_hints = filt_void_list(entity_hints)

    if len(entity_hints) > 1:
        keys_to_delete = list(entity_hints.keys())[1:]
        for key in keys_to_delete:
                del entity_hints[key]
    
    return entity_hints, first_entity

def filt_void_list(hints_list):
    filtered_hints_list = {key: value for key, value in hints_list.items() if value}
    return filtered_hints_list


if __name__ == "__main__":
    # file_path = './test.txt'
    # read_file_and_process(file_path)
    # text = '\n    ### Instruction:\n    Suppose you are a seasoned mathematician tasked with analyzing a mathematical problem. First, identify key entities relevant to solving the problem and then generate multiple event hints for each entity to uncover both explicit and implicit knowledge about the mathematical issue. This will help in forming a detailed understanding of the problem. Please ensure that your event hints are logically connected to the entities and are helpful in solving the mathematical problem.\n\n    ----\n    ### Input:\n    "Problem": "There are 87 oranges and 290 bananas in Philip\'s collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups. How big is each group of bananas?"\n    We aim to break down this problem by identifying its key entities and generating event hints that assist in solving it:\n\n\n    ### Entity Event Hints Response:\n    "Entity Event Hints":\n    **start**\n        \n        Total number of bananas:"\n        \n        1. The problem states that there are 290 bananas in total.\n        \n        "\n        \n        Number of groups for bananas:"\n        \n        1. The bananas are to be divided into 2 groups.\n        \n        "\n        \n    **ending**\n    \n    ### Input:\n    "Problem": "Edward spent $ 6 to buy 2 books each book costing him the same amount of money. Now he has $ 12. How much did each book cost?"\n    We aim to break down this problem by identifying its key entities and generating event hints that assist in solving it:\n\n\n    ### Entity Event Hints Response:\n    "Entity Event Hints":\n    **start**\n        \n        Number of books purchased:"\n        \n        1. Edward bought 2 books.\n        \n        "\n        \n        Total amount Edward has now:"\n        \n        1. Edward now has $12.\n        \n        "\n        \n        Total amount spent by Edward:"\n        \n        1. Edward spent $6 to buy the books.\n        \n        "\n        \n    **ending**\n    \n    ### Input:\n    "Problem": "Frank was reading through his favorite book. The book had 3 chapters, each with the same number of pages. It has a total of 594 pages. It took Frank 607 days to finish the book. How many pages are in each chapter?"\n    We aim to break down this problem by identifying its key entities and generating event hints that assist in solving it:\n\n\n    ### Entity Event Hints Response:\n    "Entity Event Hints":\n    **start**\n        \n        Number of chapters in the book:"\n        \n        1. The book is divided into 3 chapters.\n        \n        "\n        \n        Total number of pages in the book:"\n        \n        1. The problem states that the book has a total of 594 pages.\n        \n        "\n        \n    **ending**\n    \n    ### Input:\n    "Problem": "You have 104 dollars. How many packs of dvds can you buy if each pack costs 26 dollars?"\n    For this mathematical problem, first identify the key entities crucial for understanding and solving the problem.\n\n   After identifying the key entities, delve into the related event hints based on these entities to gain a comprehensive understanding of the problem.\n\n    ### Response:\n    "Entity Event Hints":\n    **start**\n        \n        Number of Dvds in a pack:"\n        \n        1. Each pack of dvds contains 26 dvds.\n        \n        "\n        \n        Total value of each pack:"\n        \n        1. Each pack of dvds is worth 26 dollars.\n        \n        "\n        \n    **ending**"\n    '
    # print(llama_extract_hints(text))

    # hint_list = {
    #     "Number of books in Jerry's collection": ['Jerry had 7 books in his collection.'], 
    #     "Number of action figures in Jerry's collection": ['Jerry had 3 action figures in his collection.']}
    # keys_to_delete = list(hint_list.keys())[1:]
    # for key in keys_to_delete:
    #     del hint_list[key]
    # print(hint_list)


    # 已存在的hints_list
    # hints_list = {
    #     'Number of baskets': ['There are 15 baskets.'],
    #     'Number of red peaches in each basket': ['Each basket has 19 red peaches.'],
    #     'Number of green peaches in each basket': ['Each basket has 4 green peaches.']
    # }

    # # 需要提取的关键字
    # key1 = 'Number of baskets'
    # key2 = 'Number of green peaches in each basket'

    # # 构建新的字典
    # select_list = {
    #     key1: hints_list[key1],
    #     key2: hints_list[key2]
    # }

    # # 输出结果以验证
    # print(select_list)
    # 已存在的hints_list
    hints_list = {
        'Number of baskets': ['There are 15 baskets.'],
        'Number of red peaches in each basket': ['Each basket has 19 red peaches.'],
        'Number of green peaches in each basket': ['Each basket has 4 green peaches.']
    }

    # 要添加的元素
    new_entry = {
        'Number of packs of dvds': ['The problem states that you have 104 dollars.', 'You can buy 4 packs of dvds for each 104 dollars.']
    }

    # 更新hints_list
    hints_list.update(new_entry)

    # 输出结果以验证
    print(hints_list)

