import csv
from datetime import datetime
from typing import Any, Callable

from modules.config import config
from modules.langchain.index import heavydb_index


def read_csv_file(file_name: str) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []

    with open(file_name, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)

        # Read the header (you can use or discard it depending on your needs)
        _ = next(reader)

        # Read the data rows and append them to the data list
        for row in reader:
            data.append(
                {
                    "primary_table": row[0],
                    "is_multi_table": bool(row[1] == "True"),
                    "secondary_table": row[2],
                    "question": row[3],
                }
            )

    return data


def test_table_retrieval(
    question_dicts: list[dict[str, Any]],
    func: Callable[[str], Any],
    res_parser: Callable[[Any], set[str]] = lambda x: set(x),
) -> tuple[int, int, int, int]:
    total_overshot_correct = (
        0  # returned more than the tables used to make the question, but contained the tables used to make the question
    )
    total_exactly_correct = 0  # only returned the tables used to make the question
    total_partially_correct = 0  # returned at least one of the tables used to make the question
    total_incorrect = 0  # returned none of the tables used to make the question

    for question_dict in question_dicts:
        table_set = {question_dict["primary_table"]}
        if question_dict["is_multi_table"]:
            table_set.add(question_dict["secondary_table"])
        try:
            func_result = func(question_dict["question"])
            guessed_tables = res_parser(func_result)
            if guessed_tables == table_set:
                total_exactly_correct += 1
            elif guessed_tables.intersection(table_set) == table_set:
                total_overshot_correct += 1
            elif len(guessed_tables.intersection(table_set)) > 0:
                total_partially_correct += 1
            else:
                total_incorrect += 1
        except Exception:
            total_incorrect += 1

    return total_overshot_correct, total_exactly_correct, total_partially_correct, total_incorrect


def test_stuff(question_dicts: list[dict[str, Any]]) -> tuple[str, dict]:
    simple = {
        "total_overshot_correct": 0,
        "total_exactly_correct": 0,
        "total_partially_correct": 0,
        "total_incorrect": 0,
        "time_spent": 0.0,
    }

    for question_dict in question_dicts:
        table_set = {question_dict["primary_table"]}
        if question_dict["is_multi_table"]:
            table_set.add(question_dict["secondary_table"])
        question = question_dict["question"]
        try:
            simple_start = datetime.now()
            simple_res = heavydb_index.ask_about_database(question)
            simple_end = datetime.now()
            simple["time_spent"] += (simple_end - simple_start).total_seconds()
            simple_guessed_tables = set(simple_res["tables"])
            if simple_guessed_tables == table_set:
                simple["total_exactly_correct"] += 1
            elif simple_guessed_tables.intersection(table_set) == table_set:
                simple["total_overshot_correct"] += 1
            elif len(simple_guessed_tables.intersection(table_set)) > 0:
                simple["total_partially_correct"] += 1
            else:
                simple["total_incorrect"] += 1
        except Exception:
            simple["total_incorrect"] += 1

    return ("Simple Ask", simple)


def main():
    questions_file = "sample_questions.csv"
    question_dicts = read_csv_file(questions_file)
    total_questions = len(question_dicts)

    list_of_table_retrievers = [
        (
            f"Simple search using {config.huggingface_embed_model}",
            heavydb_index.simple_search_for_table_names,
            lambda x: set(x),
        )
    ]

    for desc, func, res_parser in list_of_table_retrievers:
        print(desc)
        total_overshot_correct, total_exactly_correct, total_partially_correct, total_incorrect = test_table_retrieval(
            question_dicts, func, res_parser
        )
        print(f"Total questions: {total_questions}")
        print(f"Total exactly correct: {total_exactly_correct} ({total_exactly_correct / total_questions * 100:.2f}%)")
        print(
            f"Total overshot correct: {total_overshot_correct} ({total_overshot_correct / total_questions * 100:.2f}%)"
        )
        print(
            f"Total partially correct: {total_partially_correct} ({total_partially_correct / total_questions * 100:.2f}%)"
        )
        print(f"Total incorrect: {total_incorrect} ({total_incorrect / total_questions * 100:.2f}%)")
        print("=============================================================")

    return
    desc, result = test_stuff(question_dicts)
    print(desc)
    print(f"Total questions: {total_questions}")
    print(
        f"Total exactly correct: {result['total_exactly_correct']} ({result['total_exactly_correct'] / total_questions * 100:.2f}%)"
    )
    print(
        f"Total overshot correct: {result['total_overshot_correct']} ({result['total_overshot_correct'] / total_questions * 100:.2f}%)"
    )
    print(
        f"Total partially correct: {result['total_partially_correct']} ({result['total_partially_correct'] / total_questions * 100:.2f}%)"
    )
    print(f"Total incorrect: {result['total_incorrect']} ({result['total_incorrect'] / total_questions * 100:.2f}%)")
    print(f"Time spent per question: {result['time_spent'] / total_questions:.2f} seconds")
    print("=============================================================")


if __name__ == "__main__":
    main()
