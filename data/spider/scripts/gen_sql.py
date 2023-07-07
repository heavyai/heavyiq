import argparse
import json
from llama_cpp import Llama
import re
import sys


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Eval SQL queries against ground truth queries in HeavyDB")
    parser.add_argument("-p", "--prompts", help="Test queries json file")
    parser.add_argument("-m", "--model", help="Path to model")
    parser.add_argument("--prefix", help="Optional prefix for prompt", default=None)
    parser.add_argument("--suffix", help="Optional suffix for prompt", default=None)
    parser.add_argument("-g", "--gpu", help="Use CUDA GPU", action="store_true")
    return parser.parse_args(argv)


def getPrompts(prompts_filename):
    with open(prompts_filename, "r") as f:
        prompts = json.load(f)
        return prompts


def gen_completion(prompt, model):
    completion = model(prompt, max_tokens=200, temperature=0.0, stop=[";"])
    return completion


def main(argv):
    options = getOptions(argv)
    prompts = getPrompts(options.prompts)
    model = None
    num_gpu_layers = 100 if options.gpu else 0
    model = Llama(model_path=options.model, n_gpu_layers=num_gpu_layers, n_ctx=1280)
    eval_outputs = []
    for idx, prompt_example in enumerate(prompts):
        print(idx)
        prompt = prompt_example["instruction"]
        if options.prefix is not None:
            prompt = options.prefix + prompt
        if options.suffix is not None:
            prompt = prompt + options.suffix
        completion = gen_completion(prompt, model)
        response = completion["choices"][0]["text"]
        eval_output = {"query_id": prompt_example["query_id"], "db_id": prompt_example["db_id"], "prompt": prompt, "gold_sql_query": prompt_example["output"]}
        sql = re.sub(".*(SELECT.*;).*", r"\1", response, count=0, flags=0) + ";"
        if sql is not None:
            print(sql)
            eval_output["gen_sql_query"] = sql
        else:
            eval_output["gen_sql_query"] = None
        eval_outputs.append(eval_output)
    with open("eval_queries.json", "w") as file:
        json.dump(eval_outputs, file, indent=4)



if __name__ == "__main__":
    main(sys.argv[1:])