import json
import os
from typing import Optional, Tuple
import argparse
from utils import *
from generate_open import VERSION_NUM, SUB_LIST

import logging
import multiprocessing
from openai import OpenAI, RateLimitError, APIStatusError
from functools import partial
from typing import List, Union
from tqdm import tqdm
import backoff
import abc
from tqdm import tqdm
import re
from typing import List, Union, Iterable
import argparse
import json
import requests
from time import sleep


class PredictorBase(object):
    @abc.abstractmethod
    def generate(self, prompts: Union[List[str], str]) -> List[List[str]]:
        pass

    def perform_inference(self, batch: List[str]) -> List[str]:
        # 20240424 cot-sc is disabled please use tag v1.0 for cot-sc evaluation
        return self.post_process(self.generate(batch))

    def post_process(self, completions: List[str]):
        return completions

    @abc.abstractclassmethod
    def __str__(self) -> str:
        pass


@backoff.on_exception(backoff.constant, RateLimitError, interval=60, logger=logging.getLogger()) # 超过qps暂停60秒
def get_gpt_response(prompt, model='gpt-4-0613', max_tokens=2048, stop_tokens=[], temperature=0.0) -> str:
    global client, headers
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "stop": stop_tokens,
            "temperature": temperature
        }
        if 'o1' in model or 'o2' in model or 'o3' in model or 'o4' in model:
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": max_tokens,
                "temperature": temperature
            } 
        #result = client.request("POST", "/v1", payload, headers)#.getresponse().read()
        try:
            result = requests.request(method="POST", url=client, headers=headers, json=payload).json()
        except Exception as e:
            print(f"Request error: {e=}")
            print(f"Request error: {payload=}")
            print(f"Request error: {headers=}")
            print(f"Request error: {client=}")
            print("sleep 10s and retry")
            sleep(10.0)
            return get_gpt_response(prompt, model)
        try:
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error in parsing response: {e=}")
            print(f"Response: {result=}")
            return None
    except APIStatusError as e:
        if e.status_code in [450, 451]:  # 保时洁检测
            return "内容违规，未通过保时洁检查。"
        if e.status_code == 400:
            return "输入有误，可能是上下文长度不够"
        if e.status_code == 429:
            raise e
        if e.status_code == 408:
            raise e
        print(e.message)
        return get_gpt_response(prompt, model)


def initialize_client(openai_api_key, openai_base_url):
    global client, headers
    #client = http.client.HTTPSConnection(openai_base_url)
    client = openai_base_url
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {openai_api_key}",
    }


class OpenAIPredictor(PredictorBase):
    def __init__(self, model, nproc = 16):
        self.gpt_series = model
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.openai_base_url = os.environ["OPENAI_BASE_URL"]
        self.nproc = nproc

    def generate(self, prompts: Union[List[str], str]) -> List[str]:
        with multiprocessing.Pool(
            processes=self.nproc,
            initializer=initialize_client, initargs=(
                self.openai_api_key, self.openai_base_url)
        ) as pool:
            completions = list(tqdm(
                pool.imap(
                    partial(
                        get_gpt_response,
                        model=self.gpt_series
                    ),
                    prompts
                ), total=len(prompts), desc="Performing OpenAI predictions"
            ))
        return completions

    def __str__(self) -> str:
        return f"<OpenAIPredictor>(model={self.gpt_series})"


def generate(model_path: str, 
            dataset_path: str, 
            output_path: str, 
            version: int = 0,
            prompt: str = "raw", 
            nproc: int = 16, 
            test_part: Optional[Tuple[int, int]] = None)-> None:

    """
    Generate model's response using vLLM. 

    Parameters:
        model_path: Path to your model
        dataset_path: Path to your dataset. We provide some data in eval/input as examples. If you wat to use your own dataset, please convert the dataset to json format with a list of test samples. Each sample contains at least "answer" and "type". 
        prompt: The prompt to evaluate. We provide some propmts in prompt.json and you can add new prompts following our examples.
        output_path: Path to save the results.
        tensor_parallel_size: The number of GPUs you want to use. 
        test_part: The range of data to be processed. [low, high]. If left None, will process all the data. 
    """

    # Load test data
    with open(dataset_path, 'r') as f:
        test_dataset_list = json.load(f)
    
    exists = []

    # deal with version
    if version == -1:
        test_dataset_list_new = []
        for i in range(VERSION_NUM):
            if not os.path.exists(output_path.split(".js")[0] + f"_v{i+1}.json"):
                for item in test_dataset_list:
                    ddd = item.copy()
                    ddd['problem'] = item['problem_v'+ str(i+1)] 
                    ddd['answer'] = item['answer_v' + str(i+1)]
                    ddd['version'] = i+1
                    ddd['answer_type'] = item['answer_type_v'+str(i+1)]
                    ddd['options'] = item['options_v'+str(i+1)]
                    test_dataset_list_new.append(ddd)
                print(f"VERSION: {i+1} LOADED!")
            else:
                with open(output_path.split(".js")[0] + f"_v{i+1}.json") as f:
                    exists += json.load(f)
                print(f"VERSION: {i+1} already exists!")
        for item in test_dataset_list_new:
            for i in range(VERSION_NUM):
                try:
                    del item['problem_v'+str(i+1)]
                    del item['answer_v'+str(i+1)]
                    del item['answer_type_v'+str(i+1)]
                    del item['options_v'+str(i+1)]
                except:
                    pass
        print(f"ALL VERSIONS LOADED!")
        test_dataset_list = test_dataset_list_new
        if len(test_dataset_list) == 0:
            print(f"ALL VERSIONS ALREADY EVALUATED, WRITING TO {output_path}!")
            with open(output_path, 'w') as f:
                json.dump(exists, f, indent=4)
    else:
        for i in range(VERSION_NUM):
            if i+1 == version:
                for item in test_dataset_list:
                    item['problem'] = item['problem_v'+ str(i+1)] 
                    item['answer'] = item['answer_v' + str(i+1)]
                    item['version'] = i+1 # add version
                    item['answer_type'] = item['answer_type_v'+str(i+1)]
                    item['options'] = item['options_v'+str(i+1)]
        for item in test_dataset_list:
            for i in range(VERSION_NUM):
                del item['problem_v'+str(i+1)]
                del item['answer_v'+str(i+1)]
                del item['answer_type_v'+str(i+1)]
                del item['options_v'+str(i+1)]
        print(f"VERSION: {version} LOADED!")

    # Load prompt
    with open("prompt.json", 'r') as f:
        prompt_dict = json.load(f)[prompt]
    
    # TODO: deal with inference prompt
    test_dataset_list = make_prompt(prompt_dict, test_dataset_list)

    if test_part is not None:
        test_dataset_list = test_dataset_list[test_part[0]:test_part[1]]
    batch_size = len(test_dataset_list)

    for items in test_dataset_list:
        items["completion"] = ""
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
            item["completion"] = generated_text
            

    
    with open(output_path, 'w') as f:
        json.dump(test_dataset_list, f, indent=4)
    
    # deal with versions 
    for i in range(VERSION_NUM):
        if not os.path.exists(output_path.split(".js")[0] + f"_v{i+1}.json"):
            with open(output_path.split(".js")[0] + f"_v{i+1}.json", "w") as f: # also save different versions
                ddd = [item for item in test_dataset_list if item['version'] == i+1]
                json.dump(ddd, f, indent=4)

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

    args = parser.parse_args()
    out_dir = os.path.join(args.output_dir, args.model.split("/")[-1])
    mkdir(out_dir)

    if args.subject == "all":
        for sub in SUB_LIST:
            if args.version != -1:
                raise NotImplementedError("Only support evaluating all versions if subject == all!")
            out_fn = os.path.join(out_dir, sub + ".json")
            if not os.path.exists(out_fn):
                print(f"Start evaluating {args.model.split('/')[-1]} on {sub}---version {args.version}!")
                generate(model_path=args.model,
                         dataset_path=os.path.join("data", sub + ".json"),
                         version=args.version,
                         output_path=out_fn,
                         prompt=args.prompt,
                         nproc=args.nproc)
            else:
                print(f"{args.model.split('/')[-1]} on {sub} already exist!")
    else:
        if args.version == -1:
            out_fn = os.path.join(out_dir, args.subject + ".json")
        else:
            out_fn = os.path.join(out_dir, args.subject + "_v" + str(args.version) + ".json") 
        if not os.path.exists(out_fn):
            print(f"Start evaluating {args.model.split('/')[-1]} on {args.subject}---version {args.version}!")
            generate(model_path=args.model,
                     dataset_path=os.path.join("data", args.subject + ".json"),
                     version=args.version,
                     output_path=out_fn,
                     prompt=args.prompt,
                     nproc=args.nproc)
        else:
            print(f"{args.model.split('/')[-1]} on {args.subject}  already exist!")

