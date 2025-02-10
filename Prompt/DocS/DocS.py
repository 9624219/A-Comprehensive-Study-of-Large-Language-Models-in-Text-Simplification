import time

from Dataloader.load_newsela import load_newsela_doc
from Outputparser.ff_parser import parse_ff_output
from Utils.chat_with_ollama import chat_with_ollama
from Utils.utils import extract_docs_from_list, new_dir, new_json_file, append_to_json_file, rebuild_json_file


def gen_single_llama70b_prompt2_response(passage: str) -> str:
    prompt = f"""As a text simplification writer, your task is to simplify the given text content: restate the original text in simpler and easier to understand language without changing its meaning as much as possible. You can change paragraph or sentence structure, remove some redundant information, and replace complex and uncommon expressions with simple and common ones. It should be noted that the task of text simplification is completely different from the task of text summarization, so you need to provide a simplified parallel version based on the original text, rather than just providing a brief summary. Please return the result directly without any other information."""
    prmpt_user = f"""Raw text: {passage}
Simplified text: """
    return chat_with_ollama(promptsys=prompt,promptuser=prmpt_user)


if __name__ == '__main__':
    path = r"/home/huang/remote/Agentsimp/data/articles//"
    nums = 250
    doc_list, content_list = load_newsela_doc(path, nums)
    start = time.time()
    nums = 0
    for names, contents in zip(doc_list, content_list):
        doc_name, raw_text, ver1_text, ver2_text, ver3_text, ver4_text = extract_docs_from_list(names, contents)
        print(doc_name)
        if "spanish" in doc_name:
            continue
        nums += 1
        if nums == 210:
            break
        dir_name = r"../Results/gemma/" + doc_name + "/"
        new_dir(dir_name)
        new_json_file(dir_name + "passages.json")

        prompt2_message = gen_single_llama70b_prompt2_response(raw_text)
        print('-' * 50)
        prompt2_message = prompt2_message.strip()
        print(prompt2_message)
        print('-' * 50)

        passages_data = {
            "doc_name": doc_name,
            "raw_passage": raw_text,
            "ver1_passage": ver1_text,
            "ver2_passage": ver2_text,
            "ver3_passage": ver3_text,
            "ver4_passage": ver4_text,
            "simp_gemma2": prompt2_message
        }
        append_to_json_file(dir_name + "passages.json", passages_data)
        rebuild_json_file(dir_name + "passages.json", dir_name + "passages.json")

    end = time.time()
    print("Time:", end - start)
