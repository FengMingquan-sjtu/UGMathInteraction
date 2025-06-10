import json
import os
from typing import Optional, Tuple
import argparse


from utils import *
from generate_open import VERSION_NUM, SUB_LIST
from generate_close import OpenAIPredictor



def generate(model_path: str, 
            dataset_path: str, 
            output_path: str, 
            version: int = 0,
            prompt: str = "raw", 
            nproc: int = 16, 
            phase: int = 0,
            test_part: Optional[Tuple[int, int]] = None)-> None:

    """
    Generate model's response using vLLM. 

    Parameters:
        model_path: Path to your model
        dataset_path: Path to your dataset. We provide some data in eval/input as examples. If you wat to use your own dataset, please convert the dataset to json format with a list of test samples. Each sample contains at least "answer" and "type". 
        prompt: The prompt to evaluate. We provide some propmts in prompt.json and you can add new prompts following our examples.
        output_path: Path to save the results.
        tensor_parallel_size: The number of GPUs you want to use. 
        phase: The phase of the model. 0 for original model, 1 for human model.
        test_part: The range of data to be processed. [low, high]. If left None, will process all the data. 
    """

    # Load test data
    with open(output_path, 'r') as f:
        test_dataset_list = json.load(f)
    
    # Load prompt
    with open("prompt.json", 'r') as f:
        prompt_dict = json.load(f)[prompt]
    
    
    test_dataset_list = make_prompt(prompt_dict, test_dataset_list)  # load both problem and previous discussion
    # TODO: rewrite human model prompt. delete standard prompt
    # TODO: add chat history to prompt for both human and original model

    if test_part is not None:
        test_dataset_list = test_dataset_list[test_part[0]:test_part[1]]
    batch_size = len(test_dataset_list)

    llm = OpenAIPredictor(model=model_path, nproc=nproc)

    for index_global in range(len(test_dataset_list) // batch_size):
        # Batch Construct
        if batch_size *(index_global+2) > len(test_dataset_list):
            curr_data_list = test_dataset_list[batch_size * index_global:]
        else:
            curr_data_list = test_dataset_list[batch_size * index_global:batch_size * (index_global+1)]

        # Inference
        prompts = [item['prompt'] for item in curr_data_list]
        print(prompts[0])
        completions = llm.generate(prompts)

        # Save
        for index_in_curr_batch, output in enumerate(completions):
            generated_text = output
            index_in_data_list = index_global * batch_size + index_in_curr_batch
            item = test_dataset_list[index_in_data_list]
            if phase == 1:
                generated_text = "<begin_of_user_response>" + generated_text + "<end_of_user_response>"
            item["completion"] += generated_text
            

    with open(output_path, 'w') as f:
        json.dump(test_dataset_list, f, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', 
                        type = str,
                        help="The path of model to evaluate.",
                        default="gpt-4o-mini-2024-07-18")
    parser.add_argument('--subject',
                        type=str,
                        help="The subject to evaluate.",
                        default="all")
    parser.add_argument('--version',
                        type=int,
                        help="Random version to evaluate. -1 for all versions.",
                        default=-1)
    parser.add_argument('--prompt',
                        type=str,
                        help="The prompt template to use in evaluation.",
                        default="raw")
    parser.add_argument('--output_dir', type=str, default="./results/")
    parser.add_argument('--nproc', type=int, default=16)
    parser.add_argument('--phase', type=int, help="0 for original model, 1 for human model", default=0)
    parser.add_argument('--test_part', type=int, nargs=2, default=None,
                        help="The range of data to be processed. [low, high]. If left None, will process all the data.")

    args = parser.parse_args()
    out_dir = os.path.join(args.output_dir)
    mkdir(out_dir)

    if args.subject == "all":
        for sub in SUB_LIST:
            if args.version != -1:
                raise NotImplementedError("Only support evaluating all versions if subject == all!")
            out_fn = os.path.join(out_dir, sub + ".json")
            if os.path.exists(out_fn):
                print(f"Start evaluating {args.model.split('/')[-1]} on {sub}---version {args.version}!")
                generate(model_path=args.model,
                         dataset_path=os.path.join("data", sub + ".json"),
                         version=args.version,
                         output_path=out_fn,
                         prompt=args.prompt,
                         nproc=args.nproc,
                         phase=args.phase,
                         test_part=args.test_part)
            else:
                print(f"Please run assitant on {args.model.split('/')[-1]} on {sub} first!")
    else:
        if args.version == -1:
            out_fn = os.path.join(out_dir, args.subject + ".json")
        else:
            out_fn = os.path.join(out_dir, args.subject + "_v" + str(args.version) + ".json") 
        if os.path.exists(out_fn):
            print(f"Start evaluating {args.model.split('/')[-1]} on {args.subject}---version {args.version}!")
            generate(model_path=args.model,
                     dataset_path=os.path.join("data", args.subject + ".json"),
                     version=args.version,
                     output_path=out_fn,
                     prompt=args.prompt,
                     nproc=args.nproc,
                     phase=args.phase,
                     test_part=args.test_part)
        else:
            print(f"Please run assitant on {args.model.split('/')[-1]} on {args.subject} first!")

