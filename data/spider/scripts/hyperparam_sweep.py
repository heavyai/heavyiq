import argparse
import csv
import itertools
import json
import os
import pandas as pd
import re
import shutil
import subprocess
import sys
import yaml


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Find empty tables in HeavyDB test data")
    parser.add_argument("-e", "--experiment-name", help="Experiment Name", default="experiment")
    parser.add_argument("-p", "--hyperparameters", help="JSON file containing hyperparameters to search", default=None)
    parser.add_argument("-c", "--base-config", help="YAML file containing base configuration", default=None)
    parser.add_argument("-m", "--model-prefix", help="Model prefix name", default=None)
    return parser.parse_args(argv)


def getHyperParams(hyper_params_filename):
    hyper_params = []
    with open(hyper_params_filename, "r") as f:
        hyper_params = json.load(f)
    return hyper_params


def float_range(start, end, step):
    while start <= end:
        yield start
        start += step


def getHyperParamsSweeps(hyper_params):
    # assuming hyperparameters is a list of dictionaries
    # each containing 'name', 'min', 'max', 'step' as keys
    sweeps = []
    for hyper_param in hyper_params:
        values = list(float_range(hyper_param["min"], hyper_param["max"], hyper_param["step"]))
        sweeps.append(values)
    return sweeps


def get_losses_and_perplexities(log_file_path):
    # regex pattern to match lines containing 'Mean validation loss' and 'Validation Perplexity'
    loss_pattern = re.compile(r"Mean validation loss: (\d+\.\d+)")
    perplexity_pattern = re.compile(r"Validation Perplexity: (\d+\.\d+)")

    # Initialize lists to store the loss and perplexity values
    losses = []
    perplexities = []

    # Open the log file
    with open(log_file_path, "r") as log_file:
        for line in log_file:
            # Match the regex patterns
            loss_match = loss_pattern.search(line)
            perplexity_match = perplexity_pattern.search(line)

            # If the line contains a loss value, add it to the losses list
            if loss_match is not None:
                losses.append(float(loss_match.group(1)))

            # If the line contains a perplexity value, add it to the perplexities list
            if perplexity_match is not None:
                perplexities.append(float(perplexity_match.group(1)))

    # Zip the losses and perplexities together and return them as a list of tuples
    return list(zip(losses, perplexities))


def write_logs_to_csv(filename, output_logs):
    with open(filename, "w", newline="") as csvfile:
        fieldnames = output_logs[0].keys()
        # Create a DictWriter instance
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the rows
        for output_log in output_logs:
            writer.writerow(output_log)


def main(argv):
    print(argv)
    options = getOptions(argv)
    hyper_params = getHyperParams(options.hyperparameters)
    for idx, hyper_param in enumerate(hyper_params):
        hyper_param_name = hyper_param["name"]
        hyper_param_min = hyper_param["min"]
        hyper_param_max = hyper_param["max"]
        hyper_param_step = hyper_param["step"]

        print(
            f"Hyperparam {idx + 1}: {hyper_param_name} Min: {hyper_param_min} Max: {hyper_param_max}, Step: {hyper_param_step}"
        )
    hyper_params_sweeps = getHyperParamsSweeps(hyper_params)
    print(hyper_params_sweeps)
    arg_combinations = list(itertools.product(*hyper_params_sweeps))
    program_name = "train.py"
    log_outputs = []
    for experiment_idx, arg_values in enumerate(arg_combinations):
        model_folder = f"{options.model_prefix}_{experiment_idx}"
        model_folder = os.path.abspath(model_folder)
        if os.path.exists(model_folder) and os.path.isdir(model_folder):
            shutil.rmtree(model_folder)
        os.makedirs(model_folder)
        print(f"Made {model_folder}")
        cfg_path = f"{model_folder}/cfg.yaml"

        with open(options.base_config, "r") as yaml_read_config:
            config_data = yaml.safe_load(yaml_read_config)

        config_data["output_directory"] = model_folder
        config_data["experiment_name"] = os.path.basename(model_folder)
        hyper_param_args = []
        for arg_idx, arg_value in enumerate(arg_values):
            hyper_param_name = hyper_params[arg_idx]["name"].replace("-", "_")
            config_data["training"][hyper_param_name] = arg_value
            hyper_param_args.append({"name": hyper_param_name, "value": arg_value})

        with open(cfg_path, "w") as yaml_write_config:
            yaml.safe_dump(config_data, yaml_write_config)
        print(f"Wrote to {model_folder}")

        program_args = ["--yaml", cfg_path]
        program_args_str = " ".join(program_args)
        print(f"python3 {program_name} {program_args_str}")
        command = ["python3", program_name] + program_args
        print(command)
        result = subprocess.run(command, capture_output=True, text=True)

        log_path = f"{model_folder}/logs.log"
        losses_and_perplexities = get_losses_and_perplexities(log_path)
        print(f"Experiment {experiment_idx}")
        for hyper_param_arg in hyper_param_args:
            arg_name = hyper_param_arg["name"]
            arg_value = hyper_param_arg["value"]
            print(f"{arg_name}: {arg_value}")
        print("Perplexities")
        for eval_idx, loss_and_perplexity in enumerate(losses_and_perplexities):
            val_loss = loss_and_perplexity[0]
            val_perplexity = loss_and_perplexity[1]
            log_output = {"experiment": experiment_idx}
            for hyper_param_arg in hyper_param_args:
                arg_name = hyper_param_arg["name"]
                arg_value = hyper_param_arg["value"]
                log_output[arg_name] = arg_value
            log_output["eval_idx"] = eval_idx + 1
            log_output["val_loss"] = val_loss
            log_output["val_perplexity"] = val_perplexity
            print(f"{eval_idx}: {val_perplexity}")
            log_outputs.append(log_output)
    output_csv_filename = options.experiment_name + ".csv"
    write_logs_to_csv(output_csv_filename, log_outputs)


if __name__ == "__main__":
    main(sys.argv[1:])
