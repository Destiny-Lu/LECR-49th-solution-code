# LECR-49th-solution-code

This code repository is used to store my solution code for the LECR Kaggle data science competition. This is my first time participating in a Kaggle competition, so any errors or areas for improvement are welcome. Please feel free to provide feedback on my code.

## requirements
cuml==0.6.1.post1  
cupy==11.6.0   
datasets==2.10.1   
numpy==1.24.2   
pandas==1.5.3   
scikit_learn==1.2.2   
sentence_transformers==2.2.2   
tokenizers==0.13.2   
torch==1.13.1+cu116   
tqdm==4.64.1   
transformers==4.26.1

## Training Process
1. retriever.py
2. cluster.py
3. prepare_hard_neg.py
4. retriever_listwise.py
5. prepare_retriever_distill_data.py (Not implemented in the competition.)
6. distill_retriever.py (Not implemented in the competition.)
7. cluster_ensemble.py (Not implemented in the competition.)
8. rerank.py 
9. submit_pipeline.ipynb

Translation: The above is the implemented code section. It is worth noting that our solution is a single model. Since our team participated in the competition relatively late (15 days left), many ideas were not implemented, such as distillation and ensemble recall models (code implemented but not enough training time), and ensemble rerank models (not implemented, such as stacking and LightGBM). In the future, I will add these parts and test the LB score offline (if there is time).
