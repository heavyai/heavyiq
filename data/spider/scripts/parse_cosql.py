import argparse
import csv
import json
import sys


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Parse CoSQL data")
    parser.add_argument("-i", "--input", help="CoSQL file", default="cosql_all_info_dialogs.json")
    parser.add_argument("-o", "--output", help="Csv output file", default="cosql_all_info_dialogs.csv")
    return parser.parse_args(argv)


def read_data(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def write_flattened_results_to_csv(data, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        # Write the header
        writer.writerow(
            [
                "query_id",
                "set_id",
                "set_id_num",
                "db_id",
                "num_turns",
                "turn_id",
                "question_type",
                "question",
                "sql",
                "answer",
            ]
        )
        idx = 0
        # Write the data
        for chain in data:
            set_id = chain["set_id"]
            set_id_num = chain["set_id_num"]
            db_id = chain["db_id"]
            num_turns = chain["num_turns"]
            for turn in chain["turns"]:
                id = turn["query_id"]
                turn_id = turn["turn_id"]
                question_type = turn["question_type"]
                question = turn["question"]
                sql = turn["sql"]
                answer = turn["answer"]
                writer.writerow(
                    [id, set_id, set_id_num, db_id, num_turns, turn_id, question_type, question, sql, answer]
                )
            idx += 1


def main(argv):
    options = getOptions(argv)
    data = read_data(options.input)
    set_id_num = 0
    query_id = 0
    chains = []
    total_num_turns = 0
    for key, val in data.items():
        db_id = val["db_id"]
        chain = {}
        chain["set_id"] = key
        chain["set_id_num"] = set_id_num
        chain["db_id"] = db_id
        num_turns = len(val["turns"])
        chain["turns"] = []
        # print(val['turns'])
        # print(f"{key} -> {db_id} ({num_turns} turns)")
        current_turn = {}
        turn_idx = 0
        for idx, turn in enumerate(val["turns"]):
            utterance = turn["text"]
            labels = turn.get("label", None)
            # print(f"\t{idx}: {labels} : {utterance}")
            if labels is None or len(labels) == 0:
                current_turn["sql"] = turn.get("rawSql")
            elif "INFORM_SQL" in labels:
                current_turn["question"] = utterance
                current_turn["question_type"] = "INFORM"
            elif "INFER_SQL" in labels:
                current_turn["question"] = utterance
                current_turn["question_type"] = "INFER"
            elif "AMBIGUOUS" in labels:
                current_turn["question"] = utterance
                current_turn["question_type"] = "AMBIGUOUS"
            elif "CONFIRM_SQL" in labels:
                current_turn["answer"] = utterance
            current_turn_keys = current_turn.keys()
            if "question" in current_turn_keys and "answer" in current_turn_keys and "sql" in current_turn_keys:
                if current_turn["sql"] is not None:
                    current_turn["sql"] = current_turn["sql"].replace('"', "'")
                current_turn["query_id"] = query_id
                current_turn["turn_id"] = turn_idx
                query_id += 1
                turn_idx += 1
                chain["turns"].append(current_turn)
                current_turn = {}
        set_id_num += 1
        chain["num_turns"] = turn_idx
        total_num_turns += turn_idx

        chains.append(chain)
        # if i > 1:
        #    break
    # print(chains)
    print(total_num_turns)
    write_flattened_results_to_csv(chains, options.output)


if __name__ == "__main__":
    main(sys.argv[1:])
