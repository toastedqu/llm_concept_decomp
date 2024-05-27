# Ablation Studies
# 1) Learnable weight on top of concept classifiers
# 2) Classname in concept classifiers
# 3) Concept classifier depth

import os
import json
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.stats import mode
from datasets import load_dataset, logging
from transformers import CLIPProcessor, CLIPModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# settings
logging.disable_progress_bar()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
SEED = 2
clip_model_id = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id)
clip_model.to(device)
logit_scale = clip_model.logit_scale.exp().item()

# misc funcs
def normalize(embs):
    return embs/np.linalg.norm(embs, axis=-1, keepdims=True)

def agg_score(scores, dec_level="attr"):
    scores_df = pd.concat([df[["subclass"]+[col for col in df.columns if len(col)==1]].rename(columns=lambda x:"dec_"+x), pd.DataFrame(scores.T)], axis=1)
    if dec_level == "attr":
        return scores_df.groupby(["dec_subclass"]+["dec_"+str(i) for i in range(max_count-1)])[list(range(len(scores)))].mean().to_numpy().T
    if dec_level == "subpart":
        return scores_df.groupby(["dec_subclass"]+["dec_"+str(i) for i in range(max_count-2)])[list(range(len(scores)))].mean().to_numpy().T
    if dec_level == "part":
        return scores_df.groupby(["dec_subclass","dec_0"])[list(range(len(scores)))].mean().to_numpy().T
    if dec_level == "subclass":
        return scores_df.groupby("dec_subclass")[list(range(len(scores)))].mean().to_numpy().T

# LP funcs
def CLIP_0_shot_LP_raw():
    scaler = StandardScaler()
    X_train = scaler.fit_transform(img_embs_train)
    X_test = scaler.transform(img_embs_test)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), clf, scaler

def CLIP_0_shot_LP_label_score():
    scaler = StandardScaler()
    X_train = scaler.fit_transform(np.dot(img_embs_train, lbl_embs.T)*logit_scale)
    X_test = scaler.transform(np.dot(img_embs_test, lbl_embs.T)*logit_scale)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), clf, scaler

def CLIP_0_shot_LP_score(use_cls="none"):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(scores["train"][use_cls])
    X_test = scaler.transform(scores["test"][use_cls])
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), clf, scaler

def CLIP_0_shot_LP_score_dec(use_cls="none", dec_level="attr"):
    if dec_level == "attrval": return CLIP_0_shot_LP_score(use_cls)
    new_scores_train = agg_score(scores["train"][use_cls], dec_level)
    new_scores_test = agg_score(scores["test"][use_cls], dec_level)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(new_scores_train)
    X_test = scaler.transform(new_scores_test)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), clf, scaler

# Concept classifier funcs
def CLIP_0_shot_concept_classifier(use_cls="none", dec_level="subpart"):
    if dec_level == "attr":
        reduce = 2
    if dec_level == "subpart":
        reduce = 3
    if dec_level == "part":
        reduce = max_count
    scores_df = pd.concat([df[[str(max_count-reduce)]].rename(columns=lambda x:"dec_"+x), pd.DataFrame(scores["train"][use_cls].T)], axis=1)
    scores_df = scores_df.groupby("dec_"+str(max_count-reduce))[list(range(len(scores["train"][use_cls])))].agg(list).apply(list, axis=1)
    X_train = {key: np.array(scores_df.loc[key]) for key in scores_df.index}

    scores_df = pd.concat([df[[str(max_count-reduce)]].rename(columns=lambda x:"dec_"+x), pd.DataFrame(scores["test"][use_cls].T)], axis=1)
    scores_df = scores_df.groupby("dec_"+str(max_count-reduce))[list(range(len(scores["test"][use_cls])))].agg(list).apply(list, axis=1)
    X_test = {key: np.array(scores_df.loc[key]) for key in scores_df.index}
    
    keys = list(scores_df.index)

    y_preds = []
    part_models = {part: None for part in keys}
    part_scalers = {part: None for part in keys}
    
    for part in keys:
        if part not in X_train or part not in X_test: continue
        scaler = StandardScaler()
        X_train[part] = scaler.fit_transform(X_train[part])
        X_test[part] = scaler.transform(X_test[part])
        print(X_train[part].shape, X_test[part].shape)
        print(y_train.shape, y_test.shape)
        clf = LogisticRegression(max_iter=500, multi_class='multinomial')
        clf.fit(X_train[part], y_train)
        y_pred = clf.predict(X_test[part])
        y_preds.append(y_pred)
        part_models[part] = clf
        part_scalers[part] = scaler
    return accuracy_score(y_test, mode(np.array(y_preds))[0]), part_models, part_scalers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain', default="none", help='image dataset domain')
    args = parser.parse_args()
    domain = args.domain
    domains = os.listdir("data") if domain == "none" else [domain]

    for domain in domains:
        df = pd.read_csv(fr"results\{domain}.csv", encoding="utf-8")
        max_count = max([eval(col) for col in df.columns if len(col)==1])+1
        classnames = json.load(open(r"templates\classnames.json","r"))[domain]
        part_to_attr_map = json.load(open(r"templates\part_to_attr_map.json","r"))[domain]
        partnames = set(part_to_attr_map.keys())

        ds = load_dataset(fr"data\{domain}")
        y_train, y_test = np.array(ds["train"]["label"]), np.array(ds["test"]["label"])

        img_embs_train = normalize(np.load(fr"embeddings\img_embs_{domain}_train.npy"))
        img_embs_test = normalize(np.load(fr"embeddings\img_embs_{domain}_test.npy"))
        txt_embs = normalize(np.load(fr"embeddings\txt_embs_{domain}.npy"))
        txt_embs_domain = normalize(np.load(fr"embeddings\txt_embs_{domain}_domain.npy"))
        txt_embs_subclass = normalize(np.load(fr"embeddings\txt_embs_{domain}_subclass.npy"))
        lbl_embs = normalize(np.load(fr"embeddings\lbl_embs_{domain}.npy"))

        scores = {
            "train": {
                "none": (np.dot(txt_embs, img_embs_train.T)*logit_scale).T,
                "domain": (np.dot(txt_embs_domain, img_embs_train.T)*logit_scale).T,
                "subclass": (np.dot(txt_embs_subclass, img_embs_train.T)*logit_scale).T,
            },
            "test": {
                "none": (np.dot(txt_embs, img_embs_test.T)*logit_scale).T,
                "domain": (np.dot(txt_embs_domain, img_embs_test.T)*logit_scale).T,
                "subclass": (np.dot(txt_embs_subclass, img_embs_test.T)*logit_scale).T,
            },
        }
        print("Everything is loaded. Ready to train and store LPs.")
        
        if not os.path.exists(fr"models\{domain}"): os.mkdir(fr"models\{domain}")
        results = {}
        results["raw"], clf, scaler = CLIP_0_shot_LP_raw()
        pickle.dump(clf, open(fr"models\{domain}\raw.pkl",'wb'))
        pickle.dump(scaler, open(fr"models\{domain}\scaler_raw.pkl",'wb'))
        print("Raw saved")
        
        results["label"], clf, scaler = CLIP_0_shot_LP_label_score()
        pickle.dump(clf, open(fr"models\{domain}\label.pkl",'wb'))
        pickle.dump(scaler, open(fr"models\{domain}\scaler_label.pkl",'wb'))
        print("Label saved")
        
        use_clses = ('none', 'subclass', 'domain')
        dec_levels = ('attrval', 'attr', 'subpart', 'part')
        
        for dec_level in dec_levels:
            for use_cls in use_clses:
                name = dec_level+'_'+use_cls
                results[name], clf, scaler = CLIP_0_shot_LP_score_dec(use_cls, dec_level)
                pickle.dump(clf, open(fr"models\{domain}\{name}.pkl",'wb'))
                pickle.dump(scaler, open(fr"models\{domain}\scaler_{name}.pkl",'wb'))
                print(name + " saved")
                
                if dec_level == "subpart":
                    name = dec_level+'_'+use_cls+'_'+'concept'
                    results[name], part_models, part_scalers = CLIP_0_shot_concept_classifier(use_cls, dec_level)
                    for part in part_models:
                        pickle.dump(part_models[part], open(fr"models\{domain}\{name}_{part}.pkl",'wb'))
                        pickle.dump(part_scalers[part], open(fr"models\{domain}\scaler_{name}_{part}.pkl",'wb'))
                        print(f"{name}_{part} saved")

        json.dump(results, open(fr"results\{domain}.json",'w'))
        print("Full results saved")