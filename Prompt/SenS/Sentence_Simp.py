import ollama
from tqdm import tqdm
from openai import OpenAI

def Ollama_chat(prompt_text,system_text):
    response2 = ollama.chat(model='gemma2:2b-instruct-fp16', messages=[
        {"role": "system", "content": system_text},
        {"role": "user", "content": prompt_text}
    ], stream=False, options={
        'temperature': 0.2
    },)
    re = response2['message']['content']
    return re


Instruction= f"""Please rewrite the following complex sentence in order to make it easier to understand by non-native speakers of English. You can do so by replacing complex words with simpler synonyms (i.e. paraphrasing), deleting unimportant information (i.e. compression), and/or splitting a long complex sentence into several simpler ones. The final simplified sentence needs to be grammatical, fluent, and retain the main ideas of its original counterpart without altering its meaning."""


example_sentence = ["The West Coast blues is a type of blues music characterized by jazz and jump blues influences, strong piano-dominated sounds and jazzy guitar solos, which originated from Texas blues players relocated to California in the 1940s.",
                    "Their eyes are quite small, and their visual acuity is poor.",
                    "He settled in London, devoting himself chiefly to practical teaching."]
example_labels = ['The West Coast blues is a type of blues music with jazz and jump blues influences, strong piano-dominated sounds and jazzy guitar solos. It originated from Texas blues players who moved to California in the 1940s.',
                  'They have small eyes and poor eyesight.',
                  'He lived in London. He was a teacher.']

Examples=""
for i in range(len(example_sentence)):
    Examples+=f"""####EXAMPLE {i+1}####
Sentence: {example_sentence[i]}
Simplified Sentence: {example_labels[i]}
"""
system_text = Instruction

import pandas as pd
usecols=['Expert', 'Simple']
df = pd.read_csv('./data/MedEasi/test.csv', encoding='utf-8', usecols=usecols)
Inst_list = df['Expert'].tolist()



for x in tqdm(Inst_list):
    x = x.strip()
    prompt_text = Examples
    prompt_text += f"""
Sentence: {x}
Simplified Sentence: """
    res = Ollama_chat(prompt_text,system_text)
    print('-'*50)
    res = res.strip()
    print(res)
    print('-'*50)
    write_str = f"{res}\n"
    with open("./data/result/MedEasi_gemma2_2b","a",encoding="utf-8") as f:
        f.write(write_str)


