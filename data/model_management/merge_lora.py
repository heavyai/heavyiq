import os
import argparse
import sys

import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Merge LORA with Base Model")
    parser.add_argument("-b", "--base-model", help="Path to base model")
    parser.add_argument("-l", "--lora", help="Path to LORA")
    parser.add_argument("-o", "--output", help="Path to output")
    return parser.parse_args(argv)

def main(argv):
    options = getOptions(argv)
    assert (
	options.base_model
    ), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=huggyllama/llama-7b`"  # noqa: E501

    tokenizer = LlamaTokenizer.from_pretrained(options.base_model)

    base_model = LlamaForCausalLM.from_pretrained(
	options.base_model,
	load_in_8bit=False,
	torch_dtype=torch.float16,
	device_map={"": "cpu"},
    )

    first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()

    lora_model = PeftModel.from_pretrained(
	base_model,
	options.lora,
	device_map={"": "cpu"},
	torch_dtype=torch.float16,
    )

    lora_weight = lora_model.base_model.model.model.layers[
	0
    ].self_attn.q_proj.weight

    assert torch.allclose(first_weight_old, first_weight)

    # merge weights - new merging method from peft
    lora_model = lora_model.merge_and_unload()

    lora_model.train(False)

    # did we do anything?
    assert not torch.allclose(first_weight_old, first_weight)

    lora_model_sd = lora_model.state_dict()
    deloreanized_sd = {
	k.replace("base_model.model.", ""): v
	for k, v in lora_model_sd.items()
	if "lora" not in k
    }

    LlamaForCausalLM.save_pretrained(
	base_model, options.output, state_dict=deloreanized_sd, max_shard_size="400MB"
    )

if __name__ == "__main__":
    main(sys.argv[1:])

