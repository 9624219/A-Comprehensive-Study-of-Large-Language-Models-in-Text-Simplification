import ollama
from tqdm import tqdm
from openai import OpenAI
client_gpt = OpenAI(
    api_key = "",
    base_url = ''
)

def Ollama_chat(prompt_text,system_text):
    response2 = ollama.chat(model='gemma2:2b-instruct-fp16', messages=[
        {"role": "system", "content": system_text},
        {"role": "user", "content": prompt_text}
    ], stream=False, options={
        'temperature': 0.4
    })
    re = response2['message']['content']
    return re


Instruction= f"""####INSTRUCTION####
Given a sentence containing a complex word, you should return an ordered list of "simpler" valid substitutes for the complex word in its original context. Valid substitutes are words that are simpler than the complex word and preserve the meaning of the sentence when used as a replacement. The list of simpler words (up to a maximum of 10) should be ordered by your confidence in the prediction (best predictions first). The ordered list must not contain ties."""
example_sentence = ["A Spanish government source, however, later said that banks able to cover by themselves losses on their toxic property assets will not be forced to remove them from their books while it will be compulsory for those receiving public help.",
                    "The daily death toll in Syria has declined as the number of observers has risen, but few experts expect the U.N. plan to succeed in its entirety.",
                    "A local witness said a separate group of attackers disguised in burqas — the head-to-toe robes worn by conservative Afghan women — then tried to storm the compound."]
example_complex = ["compulsory","observers","disguised"]
example_labels = [['mandatory','required','essential','forced','important','necessary','obligatory','unavoidable'],
                  ['watchers','spectators','audience','viewers','witnesses','observers','patrons','followers','detectives','reporters','onlookers'],
                  ['concealed','dressed','hidden','camouflaged','changed','covered','disguised','masked','unrecognizable','converted','impersonated']]

Examples=""
for i in range(len(example_sentence)):
    Examples+=f"""####EXAMPLE {i+1}####
Context: {example_sentence[i]}
Complex Word: {example_complex[i]}
Valid substitutes: {";".join(example_labels[i])}
"""
system_text = Instruction


Inst_list=open("./data/tsar/tsar2022_en_test_none.tsv",encoding="utf-8").readlines()
# print(GPT_chat("hello",""))
for x in tqdm(Inst_list):
    complex_word=x.split('\t')[1]
    complex_word = complex_word.strip()
    sentence=x.split('\t')[0]
    prompt_text = Examples
    prompt_text += f"""
####OUTPUT FORMAT####
Please return valid substitutions separated by commas directly and DO NOT return any other text.
Sample Output: word1;word2;word3;word4;word5;word6;word7;word8;word9;word10
####TASK####
Context: {sentence}
Complex Word: {complex_word}
Valid substitutes: """
    res = Ollama_chat(prompt_text,system_text)
    res_list = res.split(';')
    print('-'*50)
    # res_list 每个元素去除空格
    res_list = [x.strip() for x in res_list]
    print(res_list)
    print('-'*50)
    write_str = f"{sentence}\t{complex_word}\t"+"\t".join(res_list)+'\n'
    print(write_str)
    with open("./data/result/en_gemma2_2b.tsv","a",encoding="utf-8") as f:
        f.write(write_str)




#ES
Instruction= f"""####presentar####
Dada una oración que contiene una palabra compleja, debes devolver una lista ordenada de sustitutos válidos "más simples" para la palabra compleja en su contexto original. Los sustitutos válidos son palabras más simples que la palabra compleja y preservan el significado de la oración cuando se usan como reemplazo. La lista de palabras más simples (hasta un máximo de 10) debe estar ordenada por tu confianza en la predicción (las mejores predicciones primero). La lista ordenada no debe contener empates.
"""
example_sentence = ["Además de partidos de fútbol americano, el estadio ha sido utilizado para una gran variedad de eventos, entre los que se destacan varios partidos de la selección nacional de fútbol de los Estados Unidos, y fue el hogar del ahora difunto club de la MLS, el Tampa Bay Mutiny.",
                    "El representativo chileno obtuvo una muy buena participación al conquistar los tres primeros lugares del citado certamen.",
                    "Balbo era guardaespaldas de Cesare Battisti durante las manifestaciones realizadas a favor de la guerra."]
example_complex = ["difunto","representativo","guardaespaldas"]
example_labels = [['muerto','fallecido','extinto','inexistente','finado','desaparecido','acabado','inactivo'],
                  ['representante','característico','famoso','comisionado','simbólico','simbolo','modelo','ejemplar','insigne','emblemático','portavoz','grupo','seleccionado'],
                  ['guardia','segurata','escolta','defensor','gorila','protector','guardián','seguridad','cuidador','guardaespaldas','chaperón','guardía',]]
"""

"""

Examples=""
for i in range(len(example_sentence)):
    Examples+=f"""####ejemplo {i+1}####
contexto: {example_sentence[i]}
palabra compleja: {example_complex[i]}
sustitutos válidos: {";".join(example_labels[i])}
"""
system_text = Instruction


Inst_list=open("./data/tsar/tsar2022_es_test_none.tsv",encoding="utf-8").readlines()
for x in tqdm(Inst_list):
    complex_word=x.split('\t')[1]
    complex_word = complex_word.strip()
    sentence=x.split('\t')[0]
    prompt_text = Examples
    prompt_text += f"""
####formato de salida####
Por favor, devuelve sustituciones válidas separadas por comas directamente y NO devuelvas ningún otro texto.
Salida de ejemplo: palabra1;palabra2;palabra3;palabra4;palabra5;palabra6;palabra7;palabra8;palabra9;palabra10
####TAREA####
contexto: {sentence}
palabra compleja: {complex_word}
sustitutos válidos: """
    res = Ollama_chat(prompt_text,system_text)
    res_list = res.split(';')
    print('-'*50)
    # res_list 每个元素去除空格
    res_list = [x.strip() for x in res_list]
    print(res_list)
    print('-'*50)
    write_str = f"{sentence}\t{complex_word}\t"+"\t".join(res_list)+'\n'
    print(write_str)
    with open("./data/result/es_gemma2_2b.tsv","a",encoding="utf-8") as f:
        f.write(write_str)
        
        

#PT
Instruction= f"""####INSTRUÇÃO####
Dada uma frase contendo uma palavra complexa, você deve retornar uma lista ordenada de substitutos válidos "mais simples" para a palavra complexa em seu contexto original. Substitutos válidos são palavras mais simples que a palavra complexa e preservam o significado da frase quando usadas como substituição. A lista de palavras mais simples (até um máximo de 10) deve ser ordenada por sua confiança na previsão (melhores previsões primeiro). A lista ordenada não deve conter empates.
"""
example_sentence = ["esse mecanismo é o equivalente geológico de um cobertor numa noite fria que aquece a atmosfera da terra retendo radiação do sol que de outro modo se dissiparia no espaço",
                    "perguntado se o protocolo de kyoto será ratificado no caso da eleição de um democrata para a presidência em 2008 kerry surpreendeu",
                    "quem não conseguir esgotar o armazenamento de diesel puro não pode misturar com o b2 porque o produto ficaria fora de especificação"]
example_complex = ["retendo","ratificado","esgotar"]
example_labels = [['guardando','segurando','conservando','mantendo','detendo','absorvendo','possuindo','contendo','trazendo','prendendo'],
                  ['confirmado','aprovado','validado','autenticado','firmado','estipulado','corrigido'],
                  ['acabar','esvaziar','acabar com','gastar','consumir','diminuir','zerar']]

"""

"""

Examples=""
for i in range(len(example_sentence)):
    Examples+=f"""####exemplo {i+1}####
contexto: {example_sentence[i]}
Palavra Complexa: {example_complex[i]}
Substitutos Válidos: {";".join(example_labels[i])}
"""
system_text = Instruction


Inst_list=open("./data/tsar/tsar2022_pt_test_none.tsv",encoding="utf-8").readlines()
for x in tqdm(Inst_list):
    complex_word=x.split('\t')[1]
    complex_word = complex_word.strip()
    sentence=x.split('\t')[0]
    prompt_text = Examples
    prompt_text += f"""
####FORMATO DE SAÍDA####
Retorne as substituições válidas separadas por vírgulas diretamente e NÃO retorne nenhum outro texto.
Exemplo de Saída: palavra1;palavra2;palavra3;palavra4;palavra5;palavra6;palavra7;palavra8;palavra9;palavra10;
####TAREFA####
contexto: {sentence}
Palavra Complexa: {complex_word}
Substitutos Válidos: """
    res = Ollama_chat(prompt_text,system_text)
    res_list = res.split(';')
    print('-'*50)
    # res_list 每个元素去除空格
    res_list = [x.strip() for x in res_list]
    print(res_list)
    print('-'*50)
    write_str = f"{sentence}\t{complex_word}\t"+"\t".join(res_list)+'\n'
    print(write_str)
    with open("./data/result/pt_gemma2_2b.tsv","a",encoding="utf-8") as f:
        f.write(write_str)