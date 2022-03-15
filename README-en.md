# [Yidianzixun Code Competition](https://tech.yidianzixun.com/competition/#/) Program Share (3rd place)--xDeepFM & ESMM

**Team: 八月无烦恼队--**[(Leaderboard)](https://tech.yidianzixun.com/competition/#/ranking)

## Idea

**The idea is to build a multi-task model based on xDeepFM. Click Duration (CD) produces after click, whose relation is similar to CTR and CVR. Therefore, we use ESMM model CTR and CTCD respectively, in which CTCD modeling is an auxiliary task to help promote CTR prediction. At last, output only contains prediction of CTR.**

## Code Structure

```
 ├── data # Data Preprocessing Module
 │   ├── DA.ipynb # Data preprocessing code
 │   ├── handle_title.ipynb
 ├── examples # Example Module
 │   ├── run_DeepFM.py
 │   ├── run_FM.py
 │   └── run_xDeepFM.py # run XDeepFM based model
 ├── generators
 │   ├── generator_m.py  # Dataset class for ESMM
 │   ├── generator.py # DataSet class for single task
 ├── grid_search.py  # grid search hyper-parameters
 ├── log # Log Module
 │   ├── tensorboard
 │   └── text
 ├── main_MT.py  # Main function of ESMM
 ├── main.py # Main Function of single task
 ├── models # Model Module
 │   ├── basemodel.py # Base model
 │   ├── deepfm.py
 │   ├── dnn.py
 │   ├── esmm.py
 │   ├── FM.py
 │   ├── layers # contains sequence and inputs layers
 │   └── xdeepfm.py  # XDeepFM model
 ├── README.md
 ├── run.bash # run script
 ├── submission
 │   ├── average.ipynb # average n runs 
 └── utils # Tools
     ├── candidate_generator.py
     ├── evaluation.py
     ├── selection.py
     └── utils.py
```

## Run

1. **Unzip all dataset and move to **`\data\`. Then, run DA.ipynb.
2. **Train model**

```
 bash run.bash
```

**All arguments are defined in main_MT.py。**
**The optimal group of hyper-parameters is:**

```
 batch_size=8192
 learning_rate=0.001
 epoch=1
 embedding_size=64
 lambda=0.01 # weight of multi-task loss
```

## Environments

```
 pip install -r requirements.txt
```
