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
    parser.add_argument('--max_tokens', type=int, default=256, help='max tokens')
    parser.add_argument('--save_suffix', type=str, default='example-suffix', help='save suffix')
    parser.add_argument('--sc_cnt', type=int, choices=range(1, 30), default=5, help='number of sc cnt')
    parser.add_argument('--model', type=str, default='./llama-2-7b-hf', help='model to use')
    parser.add_argument('--dataset', type=str, default='StrategyQA', help='dataset to use')
    parser.add_argument('--verbose', default=True, action='store_true', help='verbose mode')
    parser.add_argument("--split", default="SQA", type=str)
    parser.add_argument("--max_func_call", default=10, type=int)
    return parser

parser = get_parser()
args = parser.parse_args()


llama = guidance.llms.Transformers(args.model, device_map="auto", token_healing=True, torch_dtype=torch.bfloat16, caching=False, temperature=0.7)



def is_string_not_convertible_to_float(var):
    # 检查变量是否是字符串类型
    if isinstance(var, str):
        try:
            # 尝试将字符串转换为浮点数
            float(var)
            # 如果转换成功，则返回False
            return False
        except ValueError:
            # 如果转换失败（抛出ValueError），则返回True
            return True
    else:
        # 如果变量不是字符串，直接返回False
        return False


def cal_acc(args, final_outputs, samples, out_file):
    correct = 0
    final_output_dir = out_file + '/final'
    final_true_output_dir = final_output_dir + '/true'
    final_false_output_dir = final_output_dir + '/false'
    os.makedirs(final_true_output_dir, exist_ok=True)
    os.makedirs(final_false_output_dir, exist_ok=True)

    for sample, answer in zip(samples, final_outputs):
        if args.dataset == "gsm8k" or args.dataset == "svamp" or args.dataset == "AddSub":
            number_string = sample['gt'].replace(',', '')  # 删除逗号
            source_path = final_output_dir + f"/{sample['idx']}.txt"
            answer_match = str(sample['idx']) + " answer:" + str(answer) +" ground truth:" + str(number_string)
            print(answer_match)
            integer_part = answer.split('.')[0]
            flag = is_string_not_convertible_to_float(answer)
            if answer is None or flag == True:
                destination_path = final_false_output_dir + f"/{sample['idx']}.txt"
            elif float(integer_part) == float(number_string):
                correct += 1
                destination_path = final_true_output_dir + f"/{sample['idx']}.txt"
            else:
                destination_path = final_false_output_dir + f"/{sample['idx']}.txt"
        
        elif args.dataset == "StrategyQA" or args.dataset == "AQUA" or args.dataset == "CSQA":
            ground_answer = str(sample['gt'])
            source_path = final_output_dir + f"/{sample['idx']}.txt"
            answer_match = str(sample['idx']) + " answer:" + str(answer) +" ground truth:" + str(ground_answer)
            print(answer_match)
            if answer == None:
                destination_path = final_false_output_dir + f"/{sample['idx']}.txt"
            elif answer.lower() == ground_answer.lower():
                correct += 1
                destination_path = final_true_output_dir + f"/{sample['idx']}.txt"
            else:
                destination_path = final_false_output_dir + f"/{sample['idx']}.txt"

        shutil.copy(source_path, destination_path)
        print(f"File copy from {source_path} to {destination_path}")

    acc = correct/len(samples)
    str_acc = "accuracy:" + str(acc)
    print(str_acc)


SQA_entity_examples = [
    {
        "problem": "Did the Hopi Indians use a symbol that was similar to the swastika?",
        "key_entities": ["Hopi Indians", "Swastika"]
    },
    {
        "problem": "Hydrogen's atomic number squared exceeds number of Spice Girls?",
        "key_entities": ["Atomic number", "Hydrogen", "Spice Girls"]
    },
    {
        "problem": "Would a pear sink in water?",
        "key_entities":["Pear", "Water"]
    }
    # 添加更多示例
]


SQA_hints_examples = [
    {
        "problem": "Did the Hopi Indians use a symbol that was similar to the swastika?",
        "entity_event_hints": [
            {"key":"Hopi Indians", "hints": ["1. The Hopi Indians are a Native American tribe primarily located in northeastern Arizona.", "2. The Hopi culture is known for its spiritual and symbolic use of various icons and motifs in their art and religious practices.", "3. Historically, the Hopi have used symbols in their rituals that have similarities to symbols used by other cultures."]},
            {"key":"Swastika", "hints": ["1. The swastika is an ancient symbol used across various cultures worldwide, often associated with spirituality and auspiciousness.", "2. In the 20th century, the swastika was adopted by the Nazi Party in Germany, which dramatically altered its public perception.", "3. The swastika’s design, a hooked cross, appears in various forms in the art and symbolism of many ancient cultures, including the Native Americans."]}
        ]
    },
    {
        "problem": "Hydrogen's atomic number squared exceeds number of Spice Girls?",
        "entity_event_hints": [
            {"key":"Atomic number", "hints": ["1. The atomic number of an element represents the number of protons in the nucleus of an atom.", "2. It uniquely identifies an element and determines its chemical properties.", "3. For hydrogen, the atomic number is 1."]},
            {"key":"Hydrogen", "hints": ["1. Hydrogen is the first element in the periodic table with the atomic number 1.", "2. It is the most abundant chemical substance in the universe, primarily found in stars and gas giant planets.", "3. Hydrogen plays a crucial role in various industrial processes and energy production methods."]},
            {"key":"Spice Girls", "hints":["1. The Spice Girls were a British pop girl group formed in the 1990s.", "2. They achieved worldwide fame with their catchy songs and distinct personalities.", "3. The group consisted of members like Victoria Beckham, Mel B, Emma Bunton, Mel C, and Geri Halliwell."]}
        ]
    },
    {
        "problem": "Would a pear sink in water?",
        "entity_event_hints": [
            {"key":"Pear", "hints": ["1. A pear is a fruit that typically has a characteristic shape with a wider bottom and a narrower top.", "2. Pears are known for their juiciness and sweet taste, making them a popular fruit for consumption.", "3. The density and shape of a pear play a crucial role in determining whether it will sink or float in water."]},
            {"key":"Water", "hints": ["1. Water is a transparent, tasteless, odorless, and nearly colorless chemical substance that is essential for all forms of life.", "2. The density of water is approximately 1 gram per cubic centimeter, which is why objects either sink or float in it based on their density.", "3. The concept of buoyancy, governed by Archimedes' principle, explains whether an object will sink or float in water based on its density relative to water."]}
        ]
    }
]


SQA_summary_examples = [
    {
        "problem": "Did the Hopi Indians use a symbol that was similar to the swastika?",
        "entity_event_hints":[
            {"key":"Hopi Indians", "hints": ["1. The Hopi Indians are a Native American tribe primarily located in northeastern Arizona.", "2. The Hopi culture is known for its spiritual and symbolic use of various icons and motifs in their art and religious practices.", "3. Historically, the Hopi have used symbols in their rituals that have similarities to symbols used by other cultures."]},
            {"key":"Swastika", "hints":["1. The swastika is an ancient symbol used across various cultures worldwide, often associated with spirituality and auspiciousness.", "2. In the 20th century, the swastika was adopted by the Nazi Party in Germany, which dramatically altered its public perception.", "3. The swastika’s design, a hooked cross, appears in various forms in the art and symbolism of many ancient cultures, including the Native Americans."]}
        ],
        "summary_hints": [
            {"key":"Cultural Symbolism of Swastika among Hopi Indians","hints":["1. The Hopi Indians, based in northeastern Arizona, are known for their deep spiritual and symbolic traditions, utilizing icons and motifs in art and rituals.", "2. Historically, the Hopi culture incorporates symbols similar to the swastika found in other global cultures, suggesting a convergence in symbolic design.", "3. Despite the swastika's tarnished modern association with Nazism, its ancient form remains a sacred symbol in many cultures worldwide, including among the Hopi, reflecting universal spiritual significance."]}
        ]
    },
    {
        "problem": "Does New York Harbor sit on a craton without volcanic activity?",
        "entity_event_hints": [
            {"key":"New York Harbor", "hints": ["1. New York Harbor is a natural harbor located at the confluence of the Hudson River and the Atlantic Ocean.", "2. It serves as one of the largest natural harbors in the world and has been historically significant for trade and transportation.", "3. The harbor is surrounded by the metropolitan area of New York City, making it a crucial hub for economic activities."]},
            {"key":"Volcanic activity", "hints":["1. Volcanic activity refers to the phenomena associated with the eruption of molten rock, ash, and gases from the Earth's crust.", "2. It is typically concentrated along tectonic plate boundaries or hotspots, where magma from the mantle reaches the surface.", "3. Volcanic activity can lead to the formation of volcanic landforms such as mountains, calderas, and lava flows."]}
        ],
        "summary_hints": [
            {"key":"Geological Setting of New York Harbor","hints": ["1. New York Harbor, situated at the junction of the Hudson River and the Atlantic Ocean, stands as a prominent natural harbor crucial for global trade and transportation.", "2. The harbor, enveloped by the bustling metropolitan expanse of New York City, serves as a vital economic center, facilitating diverse commercial activities.", "3. New York Harbor rests on a stable craton devoid of volcanic activity, contrasting with regions characterized by tectonic plate boundaries or hotspots prone to volcanic eruptions, ensuring a secure maritime environment for trade and navigation."]}
        ] 
    },
    {
        "problem": "Are lengths measured in metres in the UK?",
        "entity_event_hints": [
            {"key":"Metres","hints": ["1. The meter (metre) is the base unit of length in the International System of Units (SI), symbolized as \"m\".", "2. It is defined as the distance traveled by light in a vacuum in 1/299,792,458 seconds.", "3. The meter is extensively used in scientific, engineering, and everyday applications due to its simplicity and coherence."]},
            {"key":"UK","hints": ["1. The United Kingdom (UK) comprises four countries: England, Scotland, Wales, and Northern Ireland.", "2. The UK has officially adopted the metric system for most measurements, including length, weight, and volume.", "3. While the metric system is predominant, some imperial units are still in use for specific purposes in the UK."]}
        ],
        "summary_hints": [
            {"key":"Metric System Usage for Length Measurement in the UK","hints": ["1. The United Kingdom, consisting of England, Scotland, Wales, and Northern Ireland, has embraced the metric system as the primary standard for measurements like length, weight, and volume.", "2. The metric system, including units like meters and centimeters, is widely utilized in the UK, aligning with global standards for measurement consistency.", "3. Despite metric system predominance, remnants of imperial units persist in certain specialized contexts within the UK, showcasing a blend of traditional and modern measurement practices."]}
        ]
    }
]

SQA_score_examples = [
    {
        "problem": "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?",
        "entity_event_hints": [
            {"key":"Revolving Door", "hints": ["1. Revolving doors are used in various public buildings to manage both the flow of people and energy efficiency by minimizing air exchange. --Score:\"0.7\"---", "2. These doors can control the rate of entry and exit, making it easier to monitor and secure entrances in busy or sensitive areas. --Score:\"0.9\"---"]},
            {"key":"Security Measure", "hints": ["1. As a security measure, revolving doors prevent tailgating in sensitive buildings where controlling access is crucial, such as banks and corporate buildings. --Score:\"1.0\"---", "2. Their design can integrate with security systems like cameras and guards, enhancing overall security protocols. --Score:\"0.9\"---"]},
            {"key":"High Security Locations", "hints":["1. Banks often use revolving doors to enhance security, preventing quick exits and controlling access. --Score:\"1.0\"---", "2. Department stores and malls use revolving doors for similar reasons, especially in urban areas where theft and security are significant concerns. --Score:\"0.8\"---", "3. The presence of revolving doors in such locations can deter theft and unauthorized access, aligning with stringent security measures. --Score:\"0.8\"---"]}
        ]
    },
    {
        "problem": "What do people aim to do at work?",
        "entity_event_hints": [
            {"key":"Goals", "hints": ["1. Goals represent the desired outcomes or achievements individuals strive for in their personal or professional lives. --Score:\"0.8\"---", "2. Setting clear goals at work helps individuals focus their efforts, measure progress, and stay motivated towards accomplishing specific targets. --Score:\"0.9\"---"]},
            {"key":"People", "hints": ["1. People are individuals who engage in various activities, including work, to achieve specific objectives or fulfill certain responsibilities. --Score:\"0.7\"---", "2. Understanding human behavior, motivations, and aspirations is crucial in determining what individuals aim to accomplish in different contexts. --Score:\"0.8\"---"]},
            {"key":"Work", "hints":["1. Work refers to the tasks, duties, or activities individuals perform to achieve specific outcomes, contribute to society, or earn a living. --Score:\"0.7\"---", "2. Different professions and workplaces have distinct goals and expectations regarding the work individuals are expected to carry out. --Score:\"0.8\"---"]}
        ]
    },
        {
        "problem": "Where would you find magazines along side many other printed works?",
        "entity_event_hints": [
            {"key":"Magazines", "hints": ["1. Magazines are periodical publications containing articles, photographs, and advertisements, often catering to specific interests or demographics. --Score:\"0.8\"---", "2. They are commonly found in places where people gather or wait, providing entertainment and information. --Score:\"0.7\"---"]},
            {"key":"Printed Works", "hints": ["1. Printed works encompass a wide range of materials, including books, newspapers, pamphlets, and magazines, produced through printing processes. --Score:\"0.9\"---", "2. These works are distributed in various locations based on their content and target audience. --Score:\"0.8\"---"]}
        ]
    }
]



SQA_COT_examples = [
    {
        "problem": "Do hamsters provide food for any animals?",
        "key_entities": ["Hamsters", "Predators"],
        "hints": ["1. Hamsters are small rodents native to regions in Europe and Asia, and they are a common prey for various predators due to their size and habitat.", "2. Predatory birds such as owls and hawks, as well as mammals like foxes, weasels, and snakes, regularly hunt and consume wild hamsters.", "3. Domestic cats may also prey on pet hamsters if they have the opportunity, reflecting their inherent predatory instincts towards small animals."],
        "COT": "Consider the natural role of hamsters in the food chain and who might rely on them as a source of nutrition.\nHamsters are prey animals.\nPrey are food for predators.\nThus, hamsters provide food for some animals.\nSo the answer is yes.",
        "Answer": "Yes"
    },
    {
        "problem": "Could Brooke Shields succeed at University of Pennsylvania?",
        "key_entities": ["Brooke Shields", "University of Pennsylvania"],
        "hints": ["1. Brooke Shields, an American actress and model, has demonstrated intelligence and academic dedication by graduating from Princeton University with a degree in French literature.", "2. The University of Pennsylvania, a prestigious Ivy League institution in Philadelphia, offers rigorous academic programs and has a history of successful alumni.", "3. Given her academic background and the supportive environment of the University of Pennsylvania, Brooke Shields is well-positioned to succeed academically at this institution."],
        "COT": "Consider Brooke Shields' academic history and the comparative academic rigor of Princeton University and the University of Pennsylvania. This should guide you toward understanding her potential success at the latter institution.\nBrooke Shields went to Princeton University.\nPrinceton University is about as academically rigorous as the University of Pennsylvania.\nThus, Brooke Shields could also succeed at the University of Pennsylvania.\nSo the answer is yes.",
        "Answer": "Yes"
    },
    {
        "problem": "Hydrogen's atomic number squared exceeds number of Spice Girls?",
        "key_entities": ["Hydrogen", "Spice Girls"],
        "hints": ["1. Hydrogen, the first element on the periodic table, has an atomic number of 1, which, when squared, results in 1.", "2. The Spice Girls, a famous British girl group from the 1990s, originally consisted of five members.", "3. Since 1 (Hydrogen's atomic number squared) is less than 5 (number of Spice Girls members), Hydrogen's atomic number squared does not exceed the number of Spice Girls members."],
        "COT": "Consider the atomic number of hydrogen and how many members were in the Spice Girls. Compare these numbers accordingly.\nHydrogen has an atomic number of 1.\n1 squared is 1.\nThere are 5 Spice Girls.\nThus, Hydrogen's atomic number squared is less than 5.\nSo the answer is no.",
        "Answer": "No"
    }
]

entity_structure_program = guidance(
    '''
    ### Instruction:
    Suppose you are a seasoned open-domain tasked with analyzing an open-domain problem. First, identify key entities relevant to solving the problem and then generate multiple event hints for each entity to uncover both explicit and implicit knowledge about the open-domain issue. This will help in forming a detailed understanding of the problem. Please ensure that your event hints are logically connected to the entities and are helpful in solving the open-domain problem.

    ----
    {{~! display the example format for the open-domain problem ~}}
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
    For this open-domain problem, first identify the key entities crucial for understanding and solving the problem.

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
    Suppose you are a seasoned open-domain tasked with analyzing a open-domain problem. First, identify key entities relevant to solving the problem and then generate multiple event hints for each entity to uncover both explicit and implicit knowledge about the open-domain issue. This will help in forming a detailed understanding of the problem.

    ----
    {{~! display the example format for the open-domain problem ~}}
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
    For this open-domain problem, first identify the key entities crucial for understanding and solving the problem.

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
    Suppose you are a seasoned open-domain tasked with analyzing a open-domain problem. First, identify key entities relevant to solving the problem and then generate multiple event hints for each entity to uncover both explicit and implicit knowledge about the open-domain issue. This will help in forming a detailed understanding of the problem. Please ensure that your event hints are logically connected to the entities and are helpful in solving the open-domain problem.

    ----
    {{~! display the example format for the open-domain problem ~}}
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
    For this open-domain problem, Just score the event hints, no summary or synthesis is needed. Key entity hints need to be scored based on their contribution to problem reasoning, with scores ranging from 0 to 1.


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
    Suppose you are a seasoned open-domain tasked with analyzing a open-domain problem. First, identify key entities relevant to solving the problem and then generate multiple event hints for each entity to uncover both explicit and implicit knowledge about the mathematical issue. This will help in forming a detailed understanding of the problem. Please ensure that your event hints are logically connected to the entities and are helpful in solving the open-domain problem.

    ----
    {{~! display the example format for the open-domain problem ~}}
    {{~#each examples}}
    ### Input:
    "Problem": "{{this.problem}}"
    We aim to to summarize and explore the given two key entities and event hints, in order to fully deduce summary event hints and hidden reasoning knowledge that can solve the open-domain problem.

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
    For this open-domain problem, summarize the given two entities into only one new entity and its event hints. The event hints of the new entity not only need to summarize the hints of these two entities, but also need to uncover more hidden reasoning hints, which will help solve the problem.

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
    Suppose you are a seasoned open-domain tasked with analyzing a open-domain problem. First, identify key entities relevant to solving the problem and then generate multiple event hints for each entity to uncover both explicit and implicit knowledge about the open-domain issue. This will help in forming a detailed understanding of the problem. Please ensure that your event hints are logically connected to the entities and are helpful in solving the open-domain problem.

    ----
    {{~! display the example format for the open-domain problem ~}}
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
    "{{this.COT}}"
    **ending**
    **Answer(Yes or No)**
    **start**
    "{{this.Answer}}"
    **ending**



    
    {{/each}}

    {{~! place the real question at the end }}
    ### Input:
    "Problem": "{{problem}}"
    For this open-domain problem, after obtaining potential key entities and event hints that may help solve the issue (note that the event hints may contain erroneous information), the next step is to understand these key entities and event hints, extract useful information, and gradually solve the open-domain problem.

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
    "{{gen "COT" temperature=temperature max_tokens=max_tokens stop='ending'}}"
    **ending**
    **Answer(Yes or No)**
    **start**
    "{{gen "Answer" temperature=temperature max_tokens=max_tokens stop='ending'}}"
    **ending**
    '''
)




def gen_entities(problem):
    out = entity_structure_program(
    examples = SQA_entity_examples,
        problem = problem,
        temperature=0.7,
        max_tokens=args.max_tokens,
        llm=llama
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
                examples = SQA_hints_examples,
                problem = problem,
                entity_list = entity_list,
                temperature=0.7,
                max_tokens=args.max_tokens,
                llm=llama
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
                examples = SQA_score_examples, 
                problem = problem,
                hints_list = tran_hints_list,
                temperature=0.7,
                max_tokens=args.max_tokens,
                llm=llama
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
                    examples = SQA_score_examples, 
                    problem = problem,
                    hints_list = trans_hints,
                    temperature=0.7,
                    max_tokens=args.max_tokens,
                    llm=llama
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
                    examples = SQA_summary_examples,
                    problem = problem,
                    hints_list = trans_select_hints,
                    temperature=0.7,
                    max_tokens=args.max_tokens,
                    llm=llama
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
                    examples = SQA_score_examples,
                    problem = problem,
                    hints_list = trans_new_hint,
                    temperature=0.7,
                    max_tokens=args.max_tokens,
                    llm=llama
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
            examples = SQA_COT_examples,
            problem =problem,
            entity_list = entity,
            hints = final_hint,
            temperature=0.7,
            max_tokens=1024,
            llm=llama
        )
        final_cot = out['COT']
        if args.dataset == "AQUA" or args.dataset == "CSQA":
            choice = out['choice']
            match = re.search(r'\b([A-E])\b', choice)
            final_answer = "None"
        elif args.dataset == "StrategyQA":
            answer = out['Answer']
            match = re.search(r'\b(Yes|No)\b', answer)
            final_answer = "None"
        else:
            match = re.search(r'\b[\d\.]+\b', choice)
            final_answer = "None"
        if match:
            final_answer = match.group(1)  # 提取"Yes"或"No"
            print("Extracted answer:", final_answer)
            if args.dataset == "CSQA":
                final_thought_output = final_thought_output + "\n\nself-consistency epoch:" + final_cot + "\n\nanswer:" + "\n\nchoice" + str(final_answer) 
            else:
                final_thought_output = final_thought_output + "\n\nself-consistency epoch:" + final_cot + "\n\nanswer:" + str(final_answer)
            flag = True
        else:
            print("No answer found")

    final_thought_output = final_thought_output + "\n\nfinal common answer:" + str(final_answer) 
    with open(final_output_file, 'w') as f:
        f.write(final_thought_output)
    return final_answer


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
    out_file = f'outputs/{model_name}/DD_temp/{args.dataset}'


    samples = []

    for question in tqdm(questions, total=len(questions)):

        if args.dataset == "gsm8k" or args.dataset == "svamp" or args.dataset == "AddSub" or args.dataset == "AQUA":
            if args.dataset == "gsm8k" or args.dataset == "svamp" or args.dataset == "AQUA":
                idx = question['idx']
            else:
                idx = question['qid']

            # parse question and answer
            question['question'] = parse_question(question, args.dataset)
            gt_cot, gt_ans = parse_math_ground_truth(question, args.dataset)
            if args.dataset == "gsm8k" or args.dataset == "svamp" or args.dataset == "AddSub":
                sample = {'idx': idx, 'question': question['question'], 'gt_cot': gt_cot, 'gt': gt_ans}
            else:
                sample = {'idx': idx, 'question': question['question'], 'gt_cot': gt_cot, 'options': question['options'], 'gt': gt_ans}

            # add remain fields
            for key in ['level', 'type', 'subject', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', 'ans_type']:
                if key in question:
                    sample[key] = question[key]
            samples.append(sample)  
        elif args.dataset == "StrategyQA" or args.dataset == "AQUA" or args.dataset == "CSQA":
            idx = question['idx']
            question['question'] = parse_question(question, args.dataset)
            if args.dataset == "StrategyQA":
                gt_ans, gt_cot = parse_sqa_ground_truth(question)
            else:
                gt_cot, gt_ans = parse_math_ground_truth(question, args.dataset)
            #gt_ans = question['answer']
            if args.dataset == "CSQA":
                sample = {'idx': idx, 'question': question['question'], 'gt_cot': gt_cot, 'options': question['options'], 'gt': gt_ans}
            else:
                sample = {'idx': idx, 'question': question['question'], 'gt_cot': gt_cot, 'gt': gt_ans}
            for key in ['level', 'type', 'subject', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', 'ans_type']:
                if key in question:
                    sample[key] = question[key]
            samples.append(sample)    


    self_con_turn = args.sc_cnt
    answer_lists = []
    for sample in tqdm(samples, desc="DD"):
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
