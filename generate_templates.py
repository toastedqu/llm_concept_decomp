import os
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
prompt_attr = """Following the same format as the examples, list out 3 to 6 common visual attribute categories for the {part} of a {domain}. Do NOT include any attribute value in your answer.

Example: the neck of a dog: ["length", "girth", "fur density"]
Example: the window glass of a car: ["opacity", "color tint", "thickness", "surface curvature", "reflectivity"]
Example: the petal of a flower: ["size", "shape", "color", "texture", "arrangement"]
Your turn: """

def count_money(input, output, model='gpt-4'):
    global money
    money += len(token_counter.encode(input))*money_table[model]['i']
    money += len(token_counter.encode(output))*money_table[model]['o']

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def reply(part, domain, model="gpt-4-0125-preview"):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt_attr.format(part=part, domain=domain)}
        ]
    ).choices[0].message.content

def get_all_parts(tree):
    parts = []
    for k,v in tree.items():
        if k == "attrs": continue
        elif k == "parts":
            parts += get_all_parts(tree[k])
        elif isinstance(v, dict):
            parts += [k]
            parts += get_all_parts(v)
    return parts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain', default="none", help='image dataset domain')
    parser.add_argument('--openai_api_key', default="YOUR_OPENAI_API_KEY", help='openai api key')
    args = parser.parse_args()
    os.environ['OPENAI_API_KEY'] = args.openai_api_key
    client = OpenAI()
    domain = args.domain
    domains = os.listdir("data") if domain == "none" else [domain]
    
    for domain in domains:
        partnames = get_all_parts(json.load(open(r"templates\tree_holders.json","r"))[domain])
        part_to_attr_map = json.load(open(r"templates\part_to_attr_map.json","r"))
        part_to_attr_map[domain] = {}
        for part in partnames:
            try:
                response = reply(part, domain)
                print(response)
                count_money(prompt_attr.format(part=part, domain=domain), response)
                if ':' in response: response = response.split(':')[1].strip()
                part_to_attr_map[domain][part] = eval(response)
            except:
                response = reply(part, domain)
                print(response)
                count_money(prompt_attr.format(part=part, domain=domain), response)
                if ':' in response: response = response.split(':')[1].strip()
                part_to_attr_map[domain][part] = eval(response)
            
        json.dump(part_to_attr_map, open(r"templates\part_to_attr_map.json","w"))
        print(f"Total cost: {money}")