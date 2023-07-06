import argparse
import json
import sys
import transformers
from transformers import LlamaTokenizer

def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Count tokens in a file")
    parser.add_argument("-p", "--prompts", help="Prompts file", default=None)
    parser.add_argument("-m", "--model", help="Model", default=None)
    return parser.parse_args(argv)

def readPrompts(prompts_filename):
    with open(prompts_filename, 'r') as prompts_file:
        prompts = json.load(prompts_file)
    return prompts

def countTokens(prompts, model):
    tokenizer = LlamaTokenizer.from_pretrained(model)
    sum_instruct_tokens = 0
    min_instruct_tokens = 999999999
    max_instruct_tokens = 0
    sum_answer_tokens = 0
    min_answer_tokens = 999999999
    max_answer_tokens = 0
    for prompt in prompts:
        instruction = prompt["instruction"]
        instruct_tokens = tokenizer.tokenize(instruction)
        num_instruct_tokens = len(instruct_tokens)
        sum_instruct_tokens += num_instruct_tokens
        if num_instruct_tokens < min_instruct_tokens:
            min_instruct_tokens = num_instruct_tokens
        if num_instruct_tokens > max_instruct_tokens:
            max_instruct_tokens = num_instruct_tokens
        answer = prompt["output"]
        answer_tokens = tokenizer.tokenize(answer)
        num_answer_tokens = len(answer_tokens)
        sum_answer_tokens += num_answer_tokens
        if num_answer_tokens < min_answer_tokens:
            min_answer_tokens = num_answer_tokens
        if num_answer_tokens > max_answer_tokens:
            max_answer_tokens = num_answer_tokens
    avg_instruct_tokens = sum_instruct_tokens * 1.0 / len(prompts)
    avg_answer_tokens = sum_answer_tokens * 1.0 / len(prompts)
    return {"avg_instruct_tokens": avg_instruct_tokens, "min_instruct_tokens": min_instruct_tokens, "max_instruct_tokens": max_instruct_tokens, "avg_answer_tokens": avg_answer_tokens, "min_answer_tokens": min_answer_tokens, "max_answer_tokens": max_answer_tokens}

def main(argv):
    options = getOptions(argv)
    prompts = readPrompts(options.prompts)
    tokens_stats = countTokens(prompts, options.model)
    print(f"Average tokens per instruction: {tokens_stats['avg_instruct_tokens']} tokens")
    print(f"Min tokens per instruction: {tokens_stats['min_instruct_tokens']} tokens")
    print(f"Max tokens per instruction: {tokens_stats['max_instruct_tokens']} tokens")
    print(f"Average tokens per answer: {tokens_stats['avg_answer_tokens']} tokens")
    print(f"Min tokens per answer: {tokens_stats['min_answer_tokens']} tokens")
    print(f"Max tokens per answer: {tokens_stats['max_answer_tokens']} tokens")

if __name__ == "__main__":
    main(sys.argv[1:])