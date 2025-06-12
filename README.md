LLM Interaction Evaluation on UGMathBench
------
## Description
Dual LLMs interactively solve math problems.

## Usage

#### Download Data
```bash
git lfs install  # Make sure git-lfs is installed (https://git-lfs.com)
git clone https://huggingface.co/datasets/UGMathBench/ugmathbench
mv ugmathbench/data/* ./data/
```

#### Workflow
**NOTE**: check the API keys and file path in each step.
```bash
bash gen_cmd.sh  #generate initial solution by a single LLM
bash judge_cmd.sh  # judge the solution
bash filter_wrong_cmd.sh  # collect inccorect problems as new dataset
bash gen_dual_cmd.sh # LLM interaction on the new dataset
bash judge_cmd.sh  # judge the solution on new dataset 
```

#### Credits
This code is a fork from UGMathBench repo https://github.com/YangLabHKUST/UGMathBench


