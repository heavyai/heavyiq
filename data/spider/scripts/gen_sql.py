import argparse
import json
from llama_cpp import Llama
import re
import sys


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Eval SQL queries against ground truth queries in HeavyDB")
    parser.add_argument("-p", "--prompts", help="Test queries json file")
    parser.add_argument("-m", "--model", help="Path to model")
    return parser.parse_args(argv)


def getPrompts(prompts_filename):
    with open(prompts_filename, "r") as f:
        prompts = json.load(f)
        return prompts


def gen_completion(prompt, model):
    completion = model(prompt, max_tokens=200, stop=[";"])
    return completion


def main(argv):
    options = getOptions(argv)
    prompts = getPrompts(options.prompts)
    model = Llama(model_path=options.model)
    for idx, prompt in enumerate(prompts):
        completion = gen_completion(prompt["instruction"], model)
        response = completion["choices"][0]["text"]
        print(response)
        sql = re.sub(".*(SELECT.*;).*", r"\1", response, count=0, flags=0)
        if sql is not None:
            print(sql)
        if idx > 10:
            break


if __name__ == "__main__":
    main(sys.argv[1:])
