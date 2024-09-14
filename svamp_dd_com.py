import guidance
import torch
import ast
import datasets
import json
import time
import numpy as np
import argparse
import random
import os
import argparse
import time
import openai
import numpy as np
import sys
import torch
import subprocess
import json
import logging
import shutil
from tqdm import tqdm


from datetime import datetime
from tqdm import tqdm
from utils.utils import set_seed, load_jsonl, save_jsonl, load_prompt, construct_prompt, construct_entity_prompt, construct_scores_prompt, construct_summary_prompt, construct_score_prompt, construct_final_prompt, construct_finalcode_prompt, construct_finalcode_prompt_byread, extract_str_entity, extract_str_hints, construct_final_prompt_byread, construct_final_prompt_combyread
from utils.dataload import load_data
from utils.parser import *
from utils.cuda_available import cuda_available
from src.entity_extraction import extract_entities_and_hints, extract_entitiy_and_hints, llama_extract_entites, llama_extract_hints, llama_extract_hint
from src.entity_score import extract_entities_and_scores, average_entities_scores, average_entity_scores, extract_entity_scores, llama_extract_scores
from src.entity_summary import llama_find_optimal
from utils.self_consistency import aggregate_final_answer

cuda_available = torch.cuda.is_available()
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(torch.__version__)
print(torch.version.cuda)
if cuda_available:
    # 打印 CUDA 设备信息
    print("CUDA 可用！")
    print("GPU 设备数量:", torch.cuda.device_count())
    print("当前使用的 GPU 设备:", torch.cuda.current_device())
    print("设备名称:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA 不可用，将使用 CPU 运行 PyTorch。")

# 设置默认设备为 CUDA，如果 CUDA 可用
device = torch.device('cuda' if cuda_available else 'cpu')


def get_parser():
    parser = argparse.ArgumentParser(description="dd")
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature')
    parser.add_argument('--max_tokens', type=int, default=128, help='max tokens')
    parser.add_argument('--save_suffix', type=str, default='example-suffix', help='save suffix')
    parser.add_argument('--sc_cnt', type=int, choices=range(1, 30), default=5, help='number of sc cnt')
    parser.add_argument('--model', type=str, default='./llama-2-7b-hf', help='model to use')
    parser.add_argument('--dataset', type=str, default='svamp', help='dataset to use')
    parser.add_argument('--verbose', default=True, action='store_true', help='verbose mode')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--max_func_call", default=10, type=int)
    return parser


parser = get_parser()
args = parser.parse_args()

guidance.llm = guidance.llms.transformers.LLaMA(args.model, device_map="auto", token_healing=True,
                                                torch_dtype=torch.bfloat16, caching=False, temperature=0.7)


def two_entity_to_list(first_entity, second_entity):
    need_keys = [] 
    need_keys.append(first_entity)
    need_keys.append(second_entity)
    return need_keys



def cal_acc(args, final_outputs, samples, out_file):
    correct = 0
    final_output_dir = out_file + '/final'
    final_true_output_dir = final_output_dir + '/true'
    final_false_output_dir = final_output_dir + '/false'
    os.makedirs(final_true_output_dir, exist_ok=True)
    os.makedirs(final_false_output_dir, exist_ok=True)

    for sample, answer in zip(samples, final_outputs):
        if args.dataset == "gsm8k" or args.dataset == "svamp":
            number_string = sample['gt'].replace(',', '')  # 删除逗号
            source_path = final_output_dir + f"/{sample['idx']}.txt"
            answer_match = str(sample['idx']) + " answer:" + str(answer) +" ground truth:" + str(number_string)
            print(answer_match)
            integer_part = answer.split('.')[0]
            if int(integer_part) == int(number_string):
                correct += 1
                destination_path = final_true_output_dir + f"/{sample['idx']}.txt"
            else:
                destination_path = final_false_output_dir + f"/{sample['idx']}.txt"
        
        elif args.dataset == "StrategyQA":
            ground_answer = str(sample['gt'])
            source_path = final_output_dir + f"/{sample['idx']}.txt"
            answer_match = str(sample['idx']) + " answer:" + str(answer) +" ground truth:" + str(ground_answer)
            print(answer_match)
            if answer.lower() == ground_answer.lower():
                correct += 1
                destination_path = final_true_output_dir + f"/{sample['idx']}.txt"
            else:
                destination_path = final_false_output_dir + f"/{sample['idx']}.txt"

        shutil.copy(source_path, destination_path)
        print(f"File copy from {source_path} to {destination_path}")



    acc = correct/len(samples)
    str_acc = "accuracy:" + str(acc)
    print(str_acc)


entity_hint_examples = [
    {
        "problem": "There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups. How big is each group of bananas?",
        "key_entities": ["Total number of bananas", "Number of groups for bananas"],
        "entity_event_hints": {
            "Total number of bananas": ["The problem states that there are 290 bananas in total."],
            "Number of groups for bananas": ["The bananas are to be divided into 2 groups."]
        }
    },
    {
        "problem": "Edward spent $ 6 to buy 2 books each book costing him the same amount of money. Now he has $ 12. How much did each book cost?",
        "key_entities": ["Number of books purchased", "Total amount Edward has now", "Total amount spent by Edward"],
        "entity_event_hints": {
            "Number of books purchased": ["Edward bought 2 books."],
            "Total amount Edward has now": ["Edward now has $12."],
            "Total amount spent by Edward": ["Edward spent $6 to buy the books."]
        }
    },
    {
        "problem": "Frank was reading through his favorite book. The book had 3 chapters, each with the same number of pages. It has a total of 594 pages. It took Frank 607 days to finish the book. How many pages are in each chapter?",
        "key_entities":["Number of chapters in the book", "Total number of pages in the book"],
        "entity_event_hints":{
            "Number of chapters in the book": ["The book is divided into 3 chapters."],
            "Total number of pages in the book": ["The problem states that the book has a total of 594 pages."]
        }

    }
    # 添加更多示例
]


hints_examples = [
    {
        "problem": "There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups. How big is each group of bananas?",
        "entity_event_hints": [
            {"key":"Total number of bananas", "hints": ["1.The problem states that there are 290 bananas in total."]},
            {"key":"Number of groups for bananas", "hints": ["1.The bananas are to be divided into 2 groups."]}
        ]
    },
    {
        "problem": "Edward spent $6 to buy 2 books each book costing him the same amount of money. Now he has $12. How much did each book cost?",
        "entity_event_hints": [
            {"key":"Number of books purchased", "hints": ["1.Edward bought 2 books."]},
            {"key":"Total amount Edward has now", "hints": ["1.Edward now has $12."]},
            {"key":"Total amount spent by Edward", "hints":["1.Edward spent $6 to buy the books."]}
        ]
    },
    {
        "problem": "Frank was reading through his favorite book. The book had 3 chapters, each with the same number of pages. It has a total of 594 pages. It took Frank 607 days to finish the book. How many pages are in each chapter?",
        "entity_event_hints": [
            {"key":"Number of chapters in the book", "hints": ["1.The book is divided into 3 chapters."]},
            {"key":"Total number of pages in the book", "hints": ["1.The problem states that the book has a total of 594 pages."]}
        ]
    }
]


score_hint_examples = [
    {
        "problem": "There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups. How big is each group of bananas?",
        "entity_event_hints": [
            {"key":"Number of groups for bananas", "hints":["1.The bananas are to be divided into 2 groups. --Score:\"0.9\"---"]},
            {"key":"Total number of bananas", "hints": ["1.The problem states that there are 290 bananas in total. --Score:\"1.0\"---"]}
        ]
    },
    {
        "problem": "Marco and his dad went strawberry picking. Marco's dad's strawberries weighed 11 pounds. If together their strawberries weighed 30 pounds. How much did Marco's strawberries weigh?",
        "entity_event_hints": [
            {"key":"Total weight of Marco's dad's strawberries", "hints":["1.The problem states that Marco's dad's strawberries weighed 11 pounds. --Score:\"1.0\"---"]},
            {"key":"Total weight of Marco's strawberries", "hints":["1.Let's denote the weight of Marco's strawberries as x. --Score:\"0.9\"---"]},
            {"key":"Total weight of both their strawberries", "hints":["1.The combined weight of Marco's and his dad's strawberries is 30 pounds. --Score:\"1.0\"---"]}
        ]
    },
        {
        "problem": "Edward spent $ 6 to buy 2 books each book costing him the same amount of money. Now he has $ 12. How much did each book cost?",
        "entity_event_hints": [
            {"key":"Number of books purchased", "hints":["1.Edward bought 2 books. --Score:\"1.0\"---"]},
            {"key":"Total amount Edward has now", "hints": ["1.Edward now has $12. --Score:\"0.9\"---"]},
            {"key":"Total amount spent by Edward","hints": ["1.Edward spent $6 to buy the books. --Score:\"0.8\"---"]}
        ]
    }
]

summary_hint_examples = [
    {
        "problem": "Dave had 15 apps on his phone. He added 71 new apps. After deleting some he had 14 left. How many more apps did he delete than he added?",
        "entity_event_hints":[
            {"key":"Number of apps Dave had after adding new apps","hints": ["1.After adding new apps, Dave had a total of 15 + 71 = 86 apps."]},
            {"key":"Number of apps Dave had after deleting some","hints": ["1.Dave had 14 apps left after deleting some."]}
        ],
        "summary_hints": [
            {"key":"Apps Management","hints":["1.Dave initially had 15 apps on his phone.","2.He added 71 new apps, resulting in a total of 86 apps.","3.After deleting some apps, he was left with 14 apps.","4.The difference between the number of apps he deleted and added can be calculated to determine how many more apps he deleted than added."]}
        ]
    },
    {
        "problem": "Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?",
        "entity_event_hints": [
            {"key":"Earnings for the week","hints": ["1.Eliza's earnings for the week include regular pay for the first 40 hours and overtime pay for the additional 5 hours worked.","2.The total earnings for the week are the sum of regular and overtime pay."]},
            {"key":"Eliza's Earnings","hints": ["1.Eliza earns $10 per hour for the first 40 hours of work and 1.2 times her regular hourly rate for overtime hours.","2.Eliza worked 45 hours this week.","3.Earnings are calculated based on regular and overtime rates for hours worked."]}
        ],
        "summary_hints": [
            {"key":"Eliza's Earnings","hints": ["1.Eliza earns $10 per hour for the first 40 hours of work and 1.2 times her regular hourly rate for overtime hours.","2.Eliza worked 45 hours this week.","3.Earnings are calculated based on regular and overtime rates for hours worked."]}
        ] 
    },
    {
        "problem": "The Razorback t-shirt shop makes $ 87 dollars off each t-shirt sold. During the Arkansas game and the Texas tech game they sold a total of 95 t-shirts. If they sold 47 t-shirts during the Arkansas game. How much money did they make from selling the t-shirts?",
        "entity_event_hints": [
            {"key":"Revenue per t-shirt","hints": ["1.The problem states that the shop makes $87 off each t-shirt sold."]},
            {"key":"Total number of t-shirts sold","hints": ["1.The problem mentions that a total of 95 t-shirts were sold during the Arkansas and Texas Tech games combined."]}
        ],
        "summary_hints": [
            {"key":"Revenue from T-shirt Sales","hints": ["1.A total of 95 t-shirts were sold during the Arkansas and Texas Tech games.","2.Each t-shirt sale generates $87 in revenue.","3.Multiplying the total number of t-shirts sold by the revenue per t-shirt gives the total revenue from selling the t-shirts."]}
        ]
    }
]


COT_examples = [
    {
        "problem": "Paul got a box of 110 crayons for his birthday. During the school year he gave 90 crayons to his friends while he lost 412 crayons. How many more crayons did he lose than those he gave to his friends?",
        "key_entities": ["Number of crayons given to friends", "Number of crayons lost", "Total number of crayons received"],
        "hints": ["1. Paul received a box with 110 crayons for his birthday.", "2. He gave 90 crayons to his friends during the school year.", "3. Paul lost 412 crayons in total.", "4. To determine the difference between the crayons lost and those given away, we subtract the number of crayons given away from the total number of crayons lost."],
        "COT": "Paul received 110 crayons for his birthday.\nHe gave 90 crayons to his friends during the school year.\nPaul lost 412 crayons in total.\nTo find out how many more crayons he lost than those he gave to his friends, we subtract the number of crayons given away from the total number of crayons lost.\nThe difference between the crayons lost and those given away is 412 - 90 = 322 crayons.",
        "answer": "322"
    },
    {
        "problem": "Winter is almost here and most animals are migrating to warmer countries. There are 8 bird families living near the mountain. If 42 bird families flew away to africa and 31 bird families flew away to asia. How many more bird families flew away to africa than those that flew away to asia?",
        "key_entities": ["Number of bird families that flew away to Africa", "Number of bird families that flew away to Asia", "Total number of bird families near the mountain"],
        "hints": ["1. There are 8 bird families living near the mountain.", "2. 42 bird families migrated to Africa, and 31 bird families migrated to Asia.", "3. The difference between the number of bird families migrating to Africa and Asia can be calculated to determine how many more families flew to Africa.", "4. The total number of bird families near the mountain is essential to calculate the migration discrepancy."],
        "COT": "There are 8 bird families living near the mountain.\n42 bird families flew away to Africa, and 31 bird families flew away to Asia.\nTo find out how many more bird families flew to Africa than to Asia, we need to calculate the difference between the two.\nThe total number of bird families near the mountain is 8.\nThe difference between the number of bird families that flew to Africa and Asia is 42 - 31 = 11.\nTherefore, 11 more bird families flew to Africa than to Asia.",
        "answer": "11"
    },
    {
        "problem": "Paul got a box of 267 crayons for his birthday. During the school year he lost 231 crayons while he gave away 308 crayons to his friends. How many more crayons did he give to his friends than those he lost?",
        "key_entities": ["Number of crayons given to friends", "Number of crayons lost", "Total number of crayons received"],
        "hints": ["1. Paul received a box containing 267 crayons for his birthday.", "2. He lost 231 crayons during the school year.", "3. He gave away 308 crayons to his friends.", "4. To find out how many more crayons he gave to his friends than those he lost, we need to calculate the difference between the number of crayons given away and the number lost."],
        "COT": "Paul received 267 crayons for his birthday.\nHe lost 231 crayons during the school year.\nHe gave away 308 crayons to his friends.\nTo find out how many more crayons he gave to his friends than those he lost, we calculate the difference between the number of crayons given away and the number lost.\nThe difference between the number of crayons given to friends and those lost is 308 - 231 = 77 crayons.",
        "answer": "77"
    }
]



# Define the guidance program
entity_structure_program = guidance(
    '''
    ### Instruction:
    Suppose you are a seasoned mathematician tasked with analyzing a mathematical problem. First, identify key entities relevant to solving the problem and then generate multiple event hints for each entity to uncover both explicit and implicit knowledge about the mathematical issue. This will help in forming a detailed understanding of the problem. Please ensure that your event hints are logically connected to the entities and are helpful in solving the mathematical problem.

    ----
    {{~! display the example format for the mathematical problem ~}}
    {{~#each examples}}
    ### Input:
    "Problem": "{{this.problem}}"
    We aim to break down this problem by identifying its key entities and generating event hints that assist in solving it:

    ### Key Entities Response:
    "Key Entities":
    **start**
        {{#each this.key_entities}}
        - {{this}}
        {{/each}}
    **ending**

    {{/each}}

    {{~! place the real question at the end }}
    ### Input:
    "Problem": "{{problem}}"
    For this mathematical problem, first identify the key entities crucial for understanding and solving the problem.

    ### Response:
    "Key Entities":"
    **start**
        {{gen "entities" temperature=temperature max_tokens=max_tokens stop='**ending**'}}"
    **ending**
    ''')

# stop_tokens = ["</s>", "---", "```output"]
hint_structure_program = guidance(
    '''
    ### Instruction:
    Suppose you are a seasoned mathematician tasked with analyzing a mathematical problem. First, identify key entities relevant to solving the problem and then generate multiple event hints for each entity to uncover both explicit and implicit knowledge about the mathematical issue. This will help in forming a detailed understanding of the problem.

    ----
    {{~! display the example format for the mathematical problem ~}}
    {{~#each examples}}
    ### Input:
    Problem: "{{this.problem}}"
    We aim to break down this problem by identifying its key entities and generating event hints that assist in solving it:

    Entity Event Hints:
    **start**
        {{#each this.entity_event_hints}}
        {{this.key}}:"
            {{#each this.hints}}
            {{this}}
            {{/each}}
        "
        {{/each}}
    **ending**

    {{/each}}

    {{! place the real question at the end }}
    ### Input:
    Problem: "{{problem}}"
    For this mathematical problem, first identify the key entities crucial for understanding and solving the problem.

    After identifying the key entities, delve into the related event hints based on these entities to gain a comprehensive understanding of the problem.

    ### Response:
    Entity Event Hints:
    **start**
        {{#each entity_list}}
        {{this}}:"{{gen "hint" temperature=temperature max_tokens=max_tokens stop='"'}}"
        {{/each}}
    **ending**"

    '''
)


score_structure_program = guidance(
    '''
    ### Instruction:
    Suppose you are a seasoned mathematician tasked with analyzing a mathematical problem. First, identify key entities relevant to solving the problem and then generate multiple event hints for each entity to uncover both explicit and implicit knowledge about the mathematical issue. This will help in forming a detailed understanding of the problem. Please ensure that your event hints are logically connected to the entities and are helpful in solving the mathematical problem.

    ----
    {{~! display the example format for the mathematical problem ~}}
    {{~#each examples}}
    ### Input:
    "Problem": "{{this.problem}}"
    We aim to score the event hints of key entities to fully reveal their priority relevance to the math problem.

    ### Hints Score Response:
    "Hints Score":
    **start**
        {{#each this.entity_event_hints}}
        {{this.key}}:
            {{#each this.hints}}
            {{this}}
            {{/each}}
        {{/each}}
    **ending**
    {{/each}}


    {{~! place the real question at the end }}
    ### Input:
    "Problem": "{{problem}}"
    For this mathematical problem, Just score the event hints, no summary or synthesis is needed. Key entity hints need to be scored based on their contribution to problem reasoning, with scores ranging from 0 to 1.


    ### Response:
    "Hints Score":
    **start**
        {{#each hints_list}}
        {{this.key}}
            {{#each this.hints}}
            {{this}} --Score:"{{gen "score" temperature=temperature max_tokens=max_tokens stop='"'}}"
            {{/each}}
        {{/each}}
    **ending**"
    '''
)


summary_structure_program = guidance(
    '''
    ### Instruction:
    Suppose you are a seasoned mathematician tasked with analyzing a mathematical problem. First, identify key entities relevant to solving the problem and then generate multiple event hints for each entity to uncover both explicit and implicit knowledge about the mathematical issue. This will help in forming a detailed understanding of the problem. Please ensure that your event hints are logically connected to the entities and are helpful in solving the mathematical problem.

    ----
    {{~! display the example format for the mathematical problem ~}}
    {{~#each examples}}
    ### Input:
    "Problem": "{{this.problem}}"
    We aim to to summarize and explore the given two key entities and event hints, in order to fully deduce summary event hints and hidden reasoning knowledge that can solve the mathematical problem.

    ### Event Hints Response:
    "Entity Event Hints":
    **start**
        {{#each this.entity_event_hints}}
        {{this.key}}:
            {{#each this.hints}}
            {{this}}
            {{/each}}
        {{/each}}
    **ending**

    ### Summary Event Hints Response:
    "Summary":
    **start**
        {{#each this.summary_hints}}
        {{this.key}}:
            {{#each this.hints}}
            {{this}}
            {{/each}}
        {{/each}}
    **ending**


    {{/each}}


    {{~! place the real question at the end }}
    ### Input:
    "Problem": "{{problem}}"
    For this mathematical problem, summarize the given two entities into only one new entity and its event hints. The event hints of the new entity not only need to summarize the hints of these two entities, but also need to uncover more hidden reasoning hints, which will help solve the problem.

    ### Event Hints Response:
    "Entity Event Hints":
    **start**
        {{#each hints_list}}
        {{this.key}}:
            {{#each this.hints}}
            {{this}}
            {{/each}}
        {{/each}}
    **ending**"

    ### Response:
    "Summary":
    **start**
        {{gen "summary" temperature=temperature max_tokens=max_tokens stop='**ending**'}}"
    **ending**"
    '''
)

COT_structure_program = guidance(
    '''
    ### Instruction:
    Suppose you are a seasoned mathematician tasked with analyzing a mathematical problem. First, identify key entities relevant to solving the problem and then generate multiple event hints for each entity to uncover both explicit and implicit knowledge about the mathematical issue. This will help in forming a detailed understanding of the problem. Please ensure that your event hints are logically connected to the entities and are helpful in solving the mathematical problem.

    ----
    {{~! display the example format for the mathematical problem ~}}
    {{~#each examples}}
    ### Input:
    Your main objective is to solve the problem based on the provided event hints.
    "Problem": "{{this.problem}}"

    "Key Entities":
    {{#each this.key_entities}}
    - {{this}}
    {{/each}}

    "Hints":
        {{#each this.hints}}
        - {{this}}
        {{/each}}

    ### Answer Response:
    **Thought**
    **start**
    Let's think step by step.
    {{this.COT}}
    **ending**
    **Answer(arabic numerals)**
    **start**
    {{this.answer}}
    **ending**



    
    {{/each}}

    {{~! place the real question at the end }}
    ### Input:
    "Problem": "{{problem}}"
    For this math problem, after obtaining potential key entities and event hints that may help solve the issue (note that the event hints may contain erroneous information), the next step is to understand these key entities and event hints, extract useful information, and gradually solve the math problem.

    "Key Entities":
    {{#each entity_list}}
    - {{this}}
    {{/each}}

    "Hints":
    {{#each hints}}
    - {{this}}
    {{/each}}

    ### Response:
    **Thought**
    **start**
    Let's think step by step.
    {{gen "COT" temperature=temperature max_tokens=max_tokens stop='ending'}}
    **ending**
    **Answer(arabic numerals)**
    **start**
    {{gen "answer" temperature=temperature max_tokens=max_tokens stop='ending'}}
    **ending**
    '''
)




COMCOT_structure_program = guidance(
    '''
    ### Instruction:
    Suppose you are a seasoned mathematician tasked with analyzing a mathematical problem. First, identify key entities relevant to solving the problem and then generate multiple event hints for each entity to uncover both explicit and implicit knowledge about the mathematical issue. This will help in forming a detailed understanding of the problem. Please ensure that your event hints are logically connected to the entities and are helpful in solving the mathematical problem.

    ----
    {{~! display the example format for the mathematical problem ~}}
    {{~#each examples}}
    ### Input:
    Your main objective is to solve the problem based on the provided event hints.
    "Problem": "{{this.problem}}"

    "Key Entities":
    {{#each this.key_entities}}
    - {{this}}
    {{/each}}

    "Hints":
        {{#each this.hints}}
        - {{this}}
        {{/each}}

    ### Answer Response:
    **Solution**:
    **start**
        **Thought**:"
        Let's think step by step.
        {{this.COT}}"
        **Answer(arabic numerals)**:"
        {{this.answer}}
    **ending**



    
    {{/each}}

    {{~! place the real question at the end }}
    ### Input:
    "Problem": "{{problem}}"
    For this math problem, after obtaining potential key entities and event hints that may help solve the issue (note that the event hints may contain erroneous information), the next step is to understand these key entities and event hints, extract useful information, and gradually solve the math problem.

    "Key Entities":
    {{#each entity_list}}
    - {{this}}
    {{/each}}

    "Hints":
    {{#each hints}}
    - {{this}}
    {{/each}}

    ### Response:
    **Solution**:
    **start**
        **Thought**:"
        Let's think step by step.
        {{gen "COT" temperature=temperature max_tokens=max_tokens stop='"'}}"
        **Answer(arabic numerals)**:"
        {{gen "answer" temperature=temperature max_tokens=max_tokens stop='"'}}"
    **ending**
    '''
)



def gen_entities(problem):
    out = entity_structure_program(
    examples = entity_hint_examples,
        problem = problem,
        temperature=0.7,
        max_tokens=args.max_tokens
    )
    str_entities = out['entities']
    entity_list = llama_extract_entites(str_entities)
    print(entity_list)
    return entity_list


def gen_hints(problem, entity_list):
    flag = True
    if len(entity_list) > 0:
        while flag:
            out = hint_structure_program(
                examples = hints_examples,
                problem = problem,
                entity_list = entity_list,
                temperature=0.7,
                max_tokens=args.max_tokens
            )
            str_hint = out.text
            entities_hints = llama_extract_hints(str_hint)
            if len(entities_hints) > 0:
                flag = False   
    else:
        entities_hints = []
    
    return entities_hints


def gen_score(problem, hints_list, entity_list):
    flag = False
    loop_count = 0
    if len(hints_list) > 0:
        tran_hints_list = [{"key": key,"hints": [f"{index + 1}. {hint}" for index, hint in enumerate(hints)]} for key, hints in hints_list.items()]
        while not flag:
            loop_count += 1
            out = score_structure_program(
                examples = score_hint_examples, 
                problem = problem,
                hints_list = tran_hints_list,
                temperature=0.7,
                max_tokens=args.max_tokens
            )
            str_scores = out.text
            flag, score_list = llama_extract_scores(str_scores, hints_list)
            if loop_count > 5:
                score_list = {}
                for entity in entity_list:
                    score_list[entity] = random.random()
                break
    else:
        score_list = []
    return score_list

def gen_summary(args, problem, score_list, hints_list, output_for_file, for_output):
    max_func_call = args.max_func_call
    final_hints = None
    if len(score_list) > 0 and len(hints_list) > 0:
        for epoch in range(max_func_call):
            if len(hints_list) == 0 and len(score_list) == 0:
                first_hint = []
                final_hints = first_hint
                break
            print("=" * 50, "Epoch", epoch)
            print(hints_list)
            print(score_list)
            if len(hints_list) == 1:
                for_output += str(hints_list)
                first_hint = next(iter(hints_list.values()))
                final_hints = first_hint
                break
            first_entity, second_entity, match = llama_find_optimal(hints_list, score_list)#这里要重写一个函数
            review_turn = 0 
            while not match:
                trans_hints = [{"key": key,"hints": [f"{index + 1}. {hint}" for index, hint in enumerate(hints)]} for key, hints in hints_list.items()]
                if review_turn >= 5:
                    break
                out = score_structure_program(
                    examples = score_hint_examples, 
                    problem = problem,
                    hints_list = trans_hints,
                    temperature=0.7,
                    max_tokens=args.max_tokens
                )
                _, score_list = llama_extract_scores(out.text, hints_list)
                first_entity, second_entity, match = llama_find_optimal(hints_list, score_list)
                review_turn += 1
        
            if review_turn >= 5:
                for_output += str(hints_list)
                final_hints = next(iter(hints_list.values()))
                break

            select_list = {
                first_entity: hints_list[first_entity],
                second_entity: hints_list[second_entity]
            }
            trans_select_hints = [{"key": key,"hints": [f"{index + 1}. {hint}" for index, hint in enumerate(hints)]} for key, hints in select_list.items()]
            summary_flag = False
            review_turn = 0 
            while not summary_flag:
                out = summary_structure_program(
                    examples = summary_hint_examples,
                    problem = problem,
                    hints_list = trans_select_hints,
                    temperature=0.7,
                    max_tokens=args.max_tokens
                )

                summary_out = out['summary']

                new_entity_hint, new_entity = llama_extract_hint(summary_out)#这里要重写一个函数
                if len(new_entity_hint) == 1 and new_entity in new_entity_hint:
                    summary_flag = True
                review_turn += 1

            for_output = for_output + str(hints_list) + "\n\n"
            
            if first_entity in hints_list and first_entity in score_list and second_entity in hints_list and second_entity in score_list:
                del hints_list[first_entity]
                del hints_list[second_entity]
                del score_list[first_entity]
                del score_list[second_entity]

            
            hints_list.update(new_entity_hint)

            if len(hints_list) == 1:
                for_output += str(hints_list)
                first_hint = next(iter(hints_list.values()))
                final_hints = first_hint
                break

            trans_new_hint = [{"key": key,"hints": [f"{index + 1}. {hint}" for index, hint in enumerate(hints)]} for key, hints in new_entity_hint.items()]
            score_flag = False
            loop_count = 0
            while not score_flag:   
                loop_count += 1     
                out = score_structure_program(
                    examples = score_hint_examples,
                    problem = problem,
                    hints_list = trans_new_hint,
                    temperature=0.7,
                    max_tokens=args.max_tokens
                )

                flag, new_score_list = llama_extract_scores(out.text, new_entity_hint)
                if new_entity not in new_score_list:
                    flag = False
                else:
                    score_list[new_entity] = new_score_list[new_entity]
                score_flag = flag
                if loop_count > 5:
                    score_list[new_entity] = random.random()
                break


    with open(output_for_file, 'w') as f:
        if final_hints == None or len(final_hints) == 0:
            f.write("No iterative hints")
        else:
            f.write(for_output)

    return final_hints


def gen_res(problem, entity, final_hint, final_output_file, final_thought_output):
    flag = False
    if len(entity) == 0:
        entity.append("No entity")
    if len(final_hint) == 0:
        final_hint.append("Let's think step by step without hints")
    while not flag:
        out = COT_structure_program(
            examples = COT_examples,
            problem =problem,
            entity_list = entity,
            hints = final_hint,
            temperature=0.7,
            max_tokens=args.max_tokens
        )
        final_cot = out['COT']
        answer = out['answer']
        match = re.search(r'\d+\.?\d*', answer)
        first_number = None
        if match:
            first_number = match.group()
            print("The first number found is:", first_number)
            flag = True
            final_thought_output = final_thought_output + "\n\nthought:" + final_cot + "\n\nanswer:" + str(first_number)
        else:
            print("No number found.")   
    final_thought_output = final_thought_output + "\n\nfinal common answer:" + str(first_number) 
    with open(final_output_file, 'w') as f:
        f.write(final_thought_output)
    return first_number


def entity_to_text(entity, out_file):
    with open(out_file, 'w') as file:
        if len(entity) == 0:
            file.write("No entity")
        for item in entity:
            file.write(item + '\n')    



def hints_to_txt(hint_list, out_file):
    with open(out_file, 'w') as file:
        if len(hint_list) == 0:
            file.write('No entity and hint')
        else:
            for title, hints in hint_list.items():
                file.write(title + ':\n')  # 写入标题
                for index, hint in enumerate(hints, start=1):
                    file.write(f"{index}. {hint}\n")  # 写入带编号的提示

def last_hints_to_txt(final_list, out_file):
    with open(out_file, 'w') as file:
        if len(final_list) == 0:   
            file.write("No final hints")
        else:
            for index, hint in enumerate(final_list, start=1):
                file.write(f"{index}. {hint}\n")    

def main(args):
    questions = load_data(args.dataset, args.split)
    model_name = args.model.split("/")[1]
    #model_name = "/".join(args.model.split("/")[-2:]) 
    out_file = f'outputs/{model_name}/DDCOM/{args.dataset}'

    samples = []

    for question in tqdm(questions, total=len(questions)):

        if args.dataset == "gsm8k" or args.dataset == "svamp":
            idx = question['idx']

            # parse question and answer
            question['question'] = parse_question(question, args.dataset)
            gt_cot, gt_ans = parse_math_ground_truth(question, args.dataset)
            sample = {'idx': idx, 'question': question['question'], 'gt_cot': gt_cot, 'gt': gt_ans}

            # add remain fields
            for key in ['level', 'type', 'subject', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', 'ans_type']:
                if key in question:
                    sample[key] = question[key]
            samples.append(sample)  
        elif args.dataset == "StrategyQA":

            idx = question['qid']
            question['question'] = parse_question(question, args.dataset)
            term = question['term']
            description = question['description']
            gt_ans = question['answer']
    
            sample = {'idx': idx, 'question': question['question'], 'term': term, 'description': description, 'gt': gt_ans}
            for key in ['level', 'type', 'subject', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', 'ans_type']:
                if key in question:
                    sample[key] = question[key]
            samples.append(sample)  

    print("dataset:", args.dataset, "samples:", len(samples))

    self_con_turn = args.sc_cnt
    answer_lists = []
    for sample in tqdm(samples, desc="DDCOM"):
        problem = sample['question']
        idx = sample['idx']
        en_dir = out_file + '/entity'
        for_dir = out_file + '/for'
        final_dir = out_file + '/final'
        hint_dir = out_file + '/hint'
        last_dir = out_file + '/last'
        os.makedirs(for_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)
        self_consistency_answer = []
        for sc_turn in range(self_con_turn):
            print("=" * 50, f"idx:{idx}\tself-consistency epoch 1-5:", sc_turn)
            ##生成实体
            en_sc_dir = en_dir + '/sc' + str(sc_turn)
            os.makedirs(en_sc_dir, exist_ok=True)
            entity_sc_file = en_sc_dir + f'/{idx}.txt'

            hint_sc_dir = hint_dir + '/sc' + str(sc_turn)
            os.makedirs(hint_sc_dir, exist_ok=True)
            hint_sc_file = hint_sc_dir + f'/{idx}.txt'    


            for_sc_dir = for_dir + '/sc' + str(sc_turn)
            os.makedirs(for_sc_dir, exist_ok=True)
            output_for_file = for_sc_dir + f'/{idx}.txt'
            for_output = f"idx:{idx}\n\nQuestion:{sample['question']}\n\n"

            last_sc_dir = last_dir + '/sc' + str(sc_turn)
            os.makedirs(last_sc_dir, exist_ok=True)
            last_sc_file = last_sc_dir + f'/{idx}.txt'
            final_thought_output = f"idx:{idx}\n\nSelf-Consistency:{sc_turn}\n\nQuestion:{sample['question']}\n\nGround:{sample['gt']}\n\nEntities: "


            final_sc_dir = final_dir + '/sc' + str(sc_turn)
            os.makedirs(final_sc_dir, exist_ok=True)
            final_sc_file = final_sc_dir + f'/{idx}.txt'


            entity_list = gen_entities(problem)
            entity_to_text(entity_list, entity_sc_file)
            hint_list = gen_hints(problem, entity_list)
            hints_to_txt(hint_list, hint_sc_file)
            score_list = gen_score(problem, hint_list, entity_list)
            final_hint = gen_summary(args, problem, score_list, hint_list, output_for_file, for_output)
            if final_hint == None:
                final_hint = []
            last_hints_to_txt(final_hint, last_sc_file)
            final_answer = gen_res(problem, entity_list, final_hint,final_sc_file, final_thought_output)
            self_consistency_answer.append(final_answer)
        
        common_answer = aggregate_final_answer(self_consistency_answer)
        answer_lists.append(common_answer)        
        final_file = final_dir + f'/{idx}.txt'
        final_output = f"idx:{idx}\n\nSelf-Consistency:{sc_turn}\n\nQuestion:{sample['question']}\n\nGround:{sample['gt']}\n\nAnswer:{common_answer}"
        with open(final_file, 'w') as f:
            f.write(final_output)
    
    cal_acc(args, answer_lists, samples, out_file)







    


if __name__ == "__main__": 
    main(args)

