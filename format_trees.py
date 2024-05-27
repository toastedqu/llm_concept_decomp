import re
import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset, Image
from transformers import CLIPProcessor, CLIPModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16

def format_tree(tree):
    temp = {}
    for k,v in tree.items():
        if k == "parts":
            temp.update(format_tree(tree[k]))
        elif isinstance(v, str):
            vs = re.split(' or |, |; ', v)
            temp[k] = {}
            for val in vs:
                if len(val.split()) < 5:
                    temp[k][val] = 0
        else:
            temp[k] = {}
            temp[k].update(format_tree(tree[k]))
    return temp

def path_to_text(row, dot_count, classname=None):
    if classname == "separate": prefix = f"A photo of a {row['subclass']} with "
    elif classname: prefix = f"A photo of a {classname} with "
    else: prefix = "A photo of "
    part, attr, val = row[dot_count-3], row[dot_count-2], row[dot_count-1]
    if attr == "other attributes":
        if len(val.split(' ')) == 1:
            suffix = f"{val} {part}"
        else:
            suffix = f"{part} of {val}"
    else:
        suffix = f"{part} of {val} {attr}"
    return prefix+suffix

def tree_to_df(domain):
    tree = json.load(open(fr"trees\{domain}.json",'r'))
    l = []
    for k,v in tree.items():
        d = {"subclass": k}
        d.update(v)
        l.append(d)
    df = pd.json_normalize(l).melt("subclass").sort_values(["subclass","variable"]).reset_index(drop=True)
    dot_count = max(df["variable"].str.count('\.'))
    df = pd.concat([df["subclass"], df["variable"].str.replace("attrs.","").str.split('.',expand=True)], axis=1)
    while df[dot_count-1].isnull().any():
        df.loc[df[dot_count-1].isnull(),list(range(dot_count))] = df.loc[df[dot_count-1].isnull(),list(range(dot_count))].shift(1,axis=1,fill_value=df.loc[df[dot_count-1].isnull(),0])
    df['text'] = df.apply(lambda x: path_to_text(x,dot_count), axis=1)
    df['text_com'] = df.apply(lambda x: path_to_text(x,dot_count,domain), axis=1)
    df['text_cls'] = df.apply(lambda x: path_to_text(x,dot_count,"separate"), axis=1)
    return df

def get_img_embs(ds, split):
    img_embs = []
    for i in tqdm(range(0, len(ds[split]), batch_size)):
        i_end = min(i+batch_size, len(ds[split]))
        img_emb = clip_model.get_image_features(clip_processor(text=None, images=ds[split][i:i_end]['image'], return_tensors='pt')['pixel_values'].to(device)).detach().cpu().numpy()
        img_embs.extend(img_emb)
    img_embs = np.vstack(img_embs)
    return img_embs

def store_all_img_embs():
    for domain in domains:
        ds = load_dataset(fr"data\{domain}").cast_column("image", Image())
        for split in ("train", "test"):
            np.save(fr"embeddings\img_embs_{domain}_{split}.npy", get_img_embs(ds, split))
            
def get_txt_embs(texts):
    txt_embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        i_end = min(i+batch_size, len(texts))
        txt_tokens = clip_processor(text=texts[i:i_end], images=None, return_tensors='pt', padding=True).to(device)
        txt_emb = clip_model.get_text_features(**txt_tokens).detach().cpu().numpy()
        txt_embs.extend(txt_emb)
    txt_emb = np.vstack(txt_embs)
    return txt_embs

def store_all_txt_embs():
    for domain in domains:
        df = pd.read_csv(fr"results\{domain}.csv", encoding="utf-8")
        np.save(fr"embeddings\txt_embs_{domain}.npy", get_txt_embs(df["text"].to_list()))
        np.save(fr"embeddings\txt_embs_{domain}_domain.npy", get_txt_embs(df["text_com"].to_list()))
        np.save(fr"embeddings\txt_embs_{domain}_subclass.npy", get_txt_embs(df["text_cls"].to_list()))
            
def store_all_lbl_embs():
    for domain in domains:
        np.save(fr"embeddings\lbl_embs_{domain}.npy", clip_model.get_text_features(**clip_processor(text=classnames[domain], images=None, return_tensors='pt', padding=True).to(device)).detach().cpu().numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain', default="none", help='image dataset domain')
    args = parser.parse_args()
    domain = args.domain
    domains = os.listdir("data") if domain == "none" else [domain]

    for domain in domains:
        classnames = json.load(open(r"templates\classnames.json","r"))[domain]
        formatted_tree = {}
        for classname in classnames:
            tree = json.load(open(fr"trees\{domain}\{classname}.json",'r'))
            formatted_tree[classname] = format_tree(tree)
        json.dump(formatted_tree, open(fr"trees\{domain}.json","w"))
    
    for domain in domains:
        tree_to_df(domain).to_csv(fr"results\{domain}.csv", index=False, encoding="utf-8")

    clip_model_id = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    clip_model = CLIPModel.from_pretrained(clip_model_id)
    clip_model.to(device)
    logit_scale = clip_model.logit_scale.exp().item()
    
    store_all_img_embs()
    store_all_txt_embs()
    store_all_lbl_embs()