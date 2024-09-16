# DSR
DSR is a plug-and-play prompting method


## Installation

```
conda create -n DS python==3.10
conda activate DS
pip install -r requirements.txt
```

## Setup OpenAI API

Please set `os.environ["OPENAI_API_KEY"]="YOUR_API_KEY"` in all python scripts.

Make sure that your device is able to connect to [OpenAI API](https://platform.openai.com/docs/api-reference). 

## Setup LLaMA2
**You must use 0.0.60 version of guidance while testing LlaMa models.**
Get your Llama2 weight on https://huggingface.co/meta-llama/Llama-2-13b-hf, set up on default directory: ./llama-2-7b-chat-hf (You can change the dir on era-cot/config.py)


## Run paper experiments
### GPT Experiment
You can use any openai model (`gpt-3.5-turbo`, `gpt-4`, etc.) as `YOUR_MODEL`.
- CoT:
`python main.py --model_name_or_path=YOUR_MODEL  --prompt_type=cot`
- Complex-CoT:
`python main.py --model_name_or_path=YOUR_MODEL  --prompt_type=comcot`
- CoT + DS:
`python main.py --model_name_or_path=YOUR_MODEL  --prompt_type=dd`

### LLaMA2 Experiment
You can use any LLaMA model as `YOUR_MODEL`.
- CoT + DS:
`python AddSub_DD.py --model=YOUR_MODEL`
`python CSQA_DD.py --model=YOUR_MODEL`
- ComplexCoT + DS
`python svamp_dd_com.py --model=YOUR_MODEL`
