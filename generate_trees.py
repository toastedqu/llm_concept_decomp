import os
import copy
import json
import tiktoken
import argparse
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

token_counter = tiktoken.get_encoding("cl100k_base")
money_table = {
    'gpt-4': {'i': 10/1000000,'o': 30/1000000},
    'gpt-3.5': {'i': 0.5/1000000, 'o': 1.5/1000000}
}
money = 0.
prompt_attr = """Provide an answer for each of the visual attributes mentioned below for the {part} of {species} {domain}. Each answer should be short, concise, and fewer than 5 words. If one attribute has various answers, list all of them.

{attr_str}
- other attributes (if any)"""

def count_money(input, output, model='gpt-4'):
    global money
    money += len(token_counter.encode(input))*money_table[model]['i']
    money += len(token_counter.encode(output))*money_table[model]['o']

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def reply(part, species, domain, attr_str, model="gpt-4-0125-preview"):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt_attr.format(part=part, species=species, domain=domain, attr_str=attr_str)}
        ]
    ).choices[0].message.content

def is_valid_ans(s):
    if not s: return False
    for w in ("non", "not ", "no ", "n/a"):
        if w in s: return False
    return True

def get_attrs(part, species):
    if not part_to_attr_map[part]: return {}
    attr_str = '\n'.join([f'- {attr}' for attr in part_to_attr_map[part]])
    attr_dict = {attr: None for attr in part_to_attr_map[part]}
    attr_keys = list(attr_dict.keys())

    response = reply(part, species, domain, attr_str)
    response = response.replace('**','')
    print(part, response)
    count_money(prompt_attr.format(part=part, species=species, domain=domain, attr_str=attr_str), response)
    
    try:
        lines = response.split("- ")
        for i,line in enumerate(lines):
            if ": " in line:
                k,v = line.strip().split(": ")
                k,v = k.lower(), v.lower().replace('.','')
            else:
                k,v = attr_keys[i], line.strip().lower().replace('.','')
            if is_valid_ans(v):
                attr_dict[k] = v
        for k in attr_keys:
            if not attr_dict[k]: del attr_dict[k]
        return attr_dict
    except:
        return get_attrs(part, species)

def fill_attrs(tree, species):
    for key in tree:
        if key != 'attrs' and key != 'parts':
            tree[key]['attrs'] = get_attrs(key, species)
            if "parts" in tree[key]: tree[key]["parts"] = fill_attrs(tree[key]["parts"], species)
    return tree

def generate_all_trees(domain):
    if not os.path.exists(fr"trees\{domain}"): os.mkdir(fr"trees\{domain}")
    for species in classnames:
        path = fr"trees\{domain}\{species}.json"
        json.dump(fill_attrs(copy.deepcopy(tree_holder), species), open(path,'w'))
        print(f"{path} has been generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain', default="none", help='image dataset domain')
    parser.add_argument('--openai_api_key', default="YOUR_OPENAI_API_KEY", help='openai api key')
    args = parser.parse_args()
    domain = args.domain
    domains = os.listdir("data") if domain == "none" else [domain]

    for domain in domains:
        os.environ['OPENAI_API_KEY'] = args.openai_api_key
        client = OpenAI()
        domain = args.domain
        classnames = json.load(open(r"templates\classnames.json","r"))[domain]
        part_to_attr_map = json.load(open(r"templates\part_to_attr_map.json","r"))[domain]
        partnames = set(part_to_attr_map.keys())
        tree_holder = json.load(open(r"templates\tree_holders.json","r"))[domain]
        
        generate_all_trees(domain)
        print(f"Total cost: {money}")