import ollama
from tqdm import tqdm
from openai import OpenAI

def Ollama_chat(prompt_text,system_text):
    response2 = ollama.chat(model='gemma2:2b-instruct-fp16', messages=[
        {"role": "system", "content": system_text},
        {"role": "user", "content": prompt_text}
    ], stream=False, options={
        'temperature': 0.8
    },)
    re = response2['message']['content']
    return re


# Direct
Instruction= f"""You are a helpful assistant that simplifies syntactic structures."""

system_text = Instruction

import pandas as pd
import json
Inst_list = []
file_path = './data/web_split/output.json'
with open(file_path, 'r', encoding='utf-8') as file:

    sentence = json.load(file)
    for x in sentence:
        Inst_list.append(x)

for x in tqdm(Inst_list):
    x = x.strip()
    #Direct
    prompt_text = f"""Rewrite the following paragraph using simple sentence structures and no clauses or conjunctions: {x} """

    res = Ollama_chat(prompt_text,system_text)
    print('-'*50)
    print(res)
    print('-'*50)
    write_str = f"{res}\n"
    new_data = {
        "complex": x,
        "simple": res
    }
    with open("./data/result/syntactic/websplit_gemma2_2b.json","a",encoding="utf-8") as f:
        json_line = json.dumps(new_data, ensure_ascii=False)
        f.write(json_line + '\n')


