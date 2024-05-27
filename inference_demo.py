import os
import PIL
import json
import torch
import pickle
import collections
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
from networkx.drawing.nx_pydot import graphviz_layout
from scipy.stats import mode
from transformers import CLIPProcessor, CLIPModel

# Prep
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model_id = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id)
clip_model.to(device)
logit_scale = clip_model.logit_scale.exp().item()
classnames_all = json.load(open(r"templates\classnames.json","r"))
part_to_attr_map_all = json.load(open(r"templates\part_to_attr_map.json","r"))

def normalize(embs):
    return embs/np.linalg.norm(embs, axis=-1, keepdims=True)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def infer(img):
    reduce = 3
    st.image(PIL.Image.open(img))
    img_emb = normalize(clip_model.get_image_features(clip_processor(text=None, images=PIL.Image.open(img), return_tensors='pt')['pixel_values'].to(device)).detach().cpu().numpy())
    scores = (np.dot(img_emb, txt_embs.T)*logit_scale)[0]
    scores_df = df.copy()
    scores_df['score'] = scores
    scores_df = scores_df.groupby(str(max_count-reduce))['score'].apply(list)
    
    X = {part: np.array(scores_df.loc[part]).reshape(1,-1) for part in scores_df.index}
    ys = {part: [] for part in scores_df.index}
    for part in scores_df.index:
        y = models[part].predict(scalers[part].transform(X[part])).item()
        prob = models[part].predict_proba(scalers[part].transform(X[part]))[0][int(y)]
        st.text("{0:30} {1:30} at probability {2}".format(part+" clf:", classnames[int(y)], round(prob,3)))
        ys[part] = [y, prob, models[part].coef_]
    y_pred_maj = mode(np.array([y[0] for y in ys.values()]))[0]
    temp_sum = {k[0]:0. for k in ys.values()}
    for y,p,_ in ys.values():
        temp_sum[y] += p
    y_pred_prob = sorted(temp_sum.items(), key=lambda x:x[1])[0][0]
    st.text("{0:30} {1:30}".format("The majority vote:", classnames[int(y_pred_maj)]+"!"))
    st.text("{0:30} {1:30}".format("The most probable vote:", classnames[int(y_pred_maj)]+"!"))
    
    for part in ys:
        ys[part][2] = ys[part][2][int(y_pred_prob)]
    
    return classnames[int(y_pred_prob)], ys

def get_score_tree(pred, ys):
    subpairs = []
    assigner = {part: 0 for part in ys}
    for _,path in pairs:
        if path[0] == pred:
            subpairs.append(path)
        elif not subpairs:
            assigner[path[-3]] += 1
        else:
            break

    score_tree = tree[pred].copy()
    
    weight_holder = {part: [] for part in ys}
    for path in subpairs:
        node = score_tree
        for k in path[1:-2]:
            node = node[k]
        node = node["attrs"]
        node = node[path[-2]]
        weight = ys[path[-3]][2][assigner[path[-3]]]
        node[path[-1]] = weight
        weight_holder[path[-3]].append(weight)
        assigner[path[-3]] += 1

    weight_holder = {part: softmax(np.array(v)).tolist() for part,v in weight_holder.items()}
    assigner = {part: 0 for part in ys}

    for path in subpairs:
        node = score_tree
        for k in path[1:-2]:
            node = node[k]
        node = node["attrs"]
        node = node[path[-2]]
        node[path[-1]] = weight_holder[path[-3]][assigner[path[-3]]]
        assigner[path[-3]] += 1
    return score_tree

def get_graph(score_tree, pred):
    graph = nx.DiGraph()
    counter = 0
    q = collections.deque([(None, pred+str(counter), pred, score_tree)])
    while q:
        prev_id, curr_id, curr_label, sub = q.popleft()
        if isinstance(sub, float):
            curr_id += f": {round(sub,3)}"
        if prev_id: graph.add_edge(prev_id, curr_id)
        if isinstance(sub, dict):
            for k,d in sub.items():
                counter += 1
                q.append((curr_id, k+str(counter), k, d))

    net = Network(
        height='1000px',
        width='100%',
        layout="hierarchical"
    )
    net.from_nx(graph)
    # net.options.layout.hierarchical.direction = 'LR'
    net.options.layout.hierarchical.sortMethod = 'directed'
    net.show_buttons()
    net.save_graph(f'pyvis_graph.html')

st.title("Intepretable Fine-Grained Image Classifier")

domain = st.selectbox('Please select the domain of your image:', os.listdir("data"), index=None, placeholder="Select domain here...")
if domain:
    df = pd.read_csv(fr"results\{domain}.csv", encoding="utf-8")
    max_count = max([eval(col) for col in df.columns if len(col)==1])+1
    classnames = classnames_all[domain]
    part_to_attr_map = part_to_attr_map_all[domain]
    partnames = set(df[str(max_count-3)].tolist())
    txt_embs = normalize(np.load(fr"embeddings\txt_embs_{domain}.npy"))
    models = {part: pickle.load(open(fr"models\{domain}\subpart_none_concept_{part}.pkl", 'rb')) for part in partnames}
    scalers = {part: pickle.load(open(fr"models\{domain}\scaler_subpart_none_concept_{part}.pkl", 'rb')) for part in partnames}
    pairs = json.load(open(fr"trees\text_map_{domain}.json","r"))["0"]
    tree = json.load(open(fr"trees\{domain}.json","r"))

img1 = st.file_uploader("Upload an image:", type=['png','jpg'])
img2 = st.text_input("or insert a URL below:", placeholder="Paste URL here...")
has_inferred = False
if img1:
    pred, ys = infer(img1)
    has_inferred = True
elif img2:
    urllib.request.urlretrieve(img2, f"temp.{img2.split('.')[-1]}")
    pred, ys = infer(f"temp.{img2.split('.')[-1]}")
    has_inferred = True

counter = 0
cluster_counter = 0
st.write("Pyvis takes around 30 sec to render the tree. If you wish to render the tree, click this buttion:")
if st.button("Render") and has_inferred:
    get_graph(get_score_tree(pred, ys), pred)
    components.html(open(f'pyvis_graph.html', 'r', encoding='utf-8').read(), height=435)
st.write("If you want to see the full results in a nested list, click this button:")
if st.button("Print results") and has_inferred:
    st.write(get_score_tree(pred, ys))