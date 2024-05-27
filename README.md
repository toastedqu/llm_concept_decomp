# LLM-based Hierarchical Concept Decomposition for Fine-grained Image Classification
Recent advancements in interpretable models for vision-language tasks have achieved competitive performance; however, their interpretability often suffers due to the reliance on unstructured text outputs from large language models (LLMs). This introduces randomness and compromises both transparency and reliability, which are essential for addressing safety issues in AI systems. We introduce \texttt{Hi-CoDe} (Hierarchical Concept Decomposition), a novel framework designed to enhance model interpretability through structured concept analysis. Our approach consists of two main components: (1) We use GPT-4 to decompose an input image into a structured hierarchy of visual concepts, thereby forming a visual concept tree. (2) We then employ an ensemble of simple linear classifiers that operate on concept-specific features derived from CLIP to perform classification. Our approach not only aligns with the performance of state-of-the-art models but also advances transparency by providing clear insights into the decision-making process and highlighting the importance of various concepts. This allows for a detailed analysis of potential failure modes and improves model compactness, therefore setting a new benchmark in interpretability without compromising the accuracy.

# Procedure
For a given domain, follow these steps:

1. Make sure your images is organized into the following structure:
```
data
    - DOMAIN_NAME
        - train
            - CLASS_NAME_1
                - IMG_NAME_11
                - IMG_NAME_12
                - ...
            - CLASS_NAME_2
            - ...
        - test
            - CLASS_NAME_1
                - IMG_NAME_11
                - IMG_NAME_12
                - ...
            - CLASS_NAME_2
            - ...
```

2. Generate a visual concept hierarchy with zero-shot prompting on GPT-4. Manually put them in a JSON file. A sample prompt is:
```
"Give a hierarchical list of all visual parts of the outer appearance of {DOMAIN_NAME}."
```

3. Generate a list of visual attributes per visual concept with GPT-4 in the hierarchy:
```
python generate_templates.py -d DOMAIN_NAME --openai_api_key YOUR_OPENAI_API_KEY
```


4. Fill out each attribute with possible values with GPT-4 and store them in a JSON tree:
```
python generate_trees.py -d DOMAIN_NAME --openai_api_key YOUR_OPENAI_API_KEY
```

5. Process the trees for efficient training & inference later:
```
python format_trees.py -d DOMAIN_NAME
```

6. Train & Evaluate:
```
python evaluate.py -d DOMAIN_NAME
```

7. Print results for all domains:
```
python print_results.py
```

# Inference Demo
Once you trained linear probes and standard scalars, you can run inference demo with any input image and see visualized results via streamlit:
```
streamlit run inference_demo.py
```