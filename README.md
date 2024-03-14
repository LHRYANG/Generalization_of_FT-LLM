###  Unveiling the Generalization Power of Fine-Tuned Large Language Models

#### 1. Train the model
```bash
bash run_train.sh
```

If you want to fine-tune the model with in-context learning, just change the **train.py** in **run_train.sh** to **train_ptune.py**

#### 2. Evaluate the model on various datasets
```bash
bash run_evaluate.sh
```
#### 3. Assess the performance
Modify the variable **prefix** in **evaluate_cross.py** then
```bash
python evaluate_cross.py
```
