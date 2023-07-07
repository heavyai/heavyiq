import argparse
import json
from llama_cpp import Llama
from transformers import LlamaForCausalLM, LlamaTokenizer
import re
import sys


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Eval SQL queries against ground truth queries in HeavyDB")
    parser.add_argument("-p", "--prompts", help="Test queries json file")
    parser.add_argument("-m", "--model", help="Path to model")
    parser.add_argument("--prefix", help="Optional prefix for prompt", default=None)
    parser.add_argument("--suffix", help="Optional suffix for prompt", default=None)
    parser.add_argument("-o", "--output", help="Output File", default="eval_queries.json")
    parser.add_argument("-g", "--gpu", help="Use CUDA GPU", action="store_true")
    parser.add_argument("-t", "--transformers", help="Use Transformers", action="store_true")
    return parser.parse_args(argv)


def getPrompts(prompts_filename):
    with open(prompts_filename, "r") as f:
        prompts = json.load(f)
        return prompts

def gen_transformers_cpp_completion(prompt, model, tokenizer, use_gpu):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if use_gpu:
        input_ids = input_ids.to('cuda')
    completion_tokens = model.generate(input_ids, max_length=1280, temperature=0.0, do_sample=False, num_beams=1, repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id)
    completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
    return completion

def gen_llama_cpp_completion(prompt, model):
    completion = model(prompt, max_tokens=256, temperature=0.0, stop=[";"])
    return completion

def main(argv):
    options = getOptions(argv)
    prompts = getPrompts(options.prompts)
    model = None
    tokenizer = None
    num_gpu_layers = 100 if options.gpu else 0
    if options.transformers:
        tokenizer = LlamaTokenizer.from_pretrained(options.model)
        model = LlamaForCausalLM.from_pretrained(options.model, device_map='auto')
    else:
        model = Llama(model_path=options.model, n_gpu_layers=num_gpu_layers, n_ctx=1280)
    eval_outputs = []
    for idx, prompt_example in enumerate(prompts):
        print(idx)
        prompt = prompt_example["instruction"]
        if options.prefix is not None:
            prompt = options.prefix + prompt
        if options.suffix is not None:
            prompt = prompt + options.suffix
        response = None
        if options.transformers:
            completion = gen_transformers_cpp_completion(prompt, model, tokenizer, options.gpu)
            response = completion
        else:
            completion = gen_llama_cpp_completion(prompt, model)
            response = completion["choices"][0]["text"]
        eval_output = {"query_id": prompt_example["query_id"], "db_id": prompt_example["db_id"], "prompt": prompt, "gold_sql_query": prompt_example["output"]}
        sql = re.sub(".*(SELECT.*;).*", r"\1", response, count=0, flags=0)
        if not options.transformers:
            sql += ";"
        if sql is not None:
            print(sql)
            eval_output["gen_sql_query"] = sql
        else:
            eval_output["gen_sql_query"] = None
        eval_outputs.append(eval_output)
    with open(options.output, "w") as file:
        json.dump(eval_outputs, file, indent=4)

if __name__ == "__main__":
    main(sys.argv[1:])
