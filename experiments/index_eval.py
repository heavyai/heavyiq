import sys
import asyncio
import aiofiles
from enum import Enum, auto
from aiocsv.readers import AsyncReader
from typing import Any, Callable

from heavyiq.config import get_config
from heavyiq.langchain.index import aget_heavydb_index


async def read_csv_file(file_name: str) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []

    async with aiofiles.open(file_name, mode="r", newline="", encoding="utf-8") as csvfile:
        reader = AsyncReader(csvfile)

        # Read the header (you can use or discard it depending on your needs)
        _ = await anext(reader)

        # Read the data rows and append them to the data list
        async for row in reader:
            data.append(
                {
                    "primary_table": row[0],
                    "is_multi_table": bool(row[1] == "True"),
                    "secondary_table": row[2],
                    "question": row[3],
                }
            )

    return data


class QuestionStatus(Enum):
    EXACTLY_CORRECT = auto()
    OVERSHOT_CORRECT = auto()
    PARTIALLY_CORRECT = auto()
    INCORRECT = auto()


async def process_question(
    question_dict: dict[str, Any],
    func: Callable[[str], Any],
    res_parser: Callable[[Any], set[str]] = lambda x: set(x),
) -> QuestionStatus:
    """
    Process each question to find out the given tables exists on the guessed tables list.

    Args:
        primary_table (str): primary table
        question (str): NL question.
        func (Callable): coroutine function which processes the question.
        secondary_table (str | None, optional): secondary table if there is any. Defaults to None.
    """
    table_set = {question_dict["primary_table"]}
    if question_dict["is_multi_table"]:
        table_set.add(question_dict["secondary_table"])
    try:
        func_result = await func(question_dict["question"])
        guessed_tables = res_parser(func_result)
        if guessed_tables == table_set:
            return QuestionStatus.EXACTLY_CORRECT
        elif guessed_tables.intersection(table_set) == table_set:
            return QuestionStatus.OVERSHOT_CORRECT
        elif len(guessed_tables.intersection(table_set)) > 0:
            return QuestionStatus.PARTIALLY_CORRECT
        else:
            return QuestionStatus.INCORRECT
    except Exception:
        return QuestionStatus.INCORRECT


async def test_table_retrieval(
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

    tasks = []
    for question_dict in question_dicts:
        tasks.append(
            asyncio.create_task(process_question(question_dict=question_dict, func=func, res_parser=res_parser))
        )

    # process each task concurrently and populate the result variable with the earliest next result
    # ie. whichever task completed first, it's result gets populated on the result variable
    for task in asyncio.as_completed(tasks):
        earliest_result: QuestionStatus = await task
        if earliest_result == QuestionStatus.EXACTLY_CORRECT:
            total_exactly_correct += 1
        elif earliest_result == QuestionStatus.OVERSHOT_CORRECT:
            total_overshot_correct += 1
        elif earliest_result == QuestionStatus.PARTIALLY_CORRECT:
            total_partially_correct += 1
        elif earliest_result == QuestionStatus.INCORRECT:
            total_incorrect += 1

    return total_overshot_correct, total_exactly_correct, total_partially_correct, total_incorrect


async def main(questions_file: str):
    question_dicts = await read_csv_file(questions_file)
    total_questions = len(question_dicts)
    heavydb_index = await aget_heavydb_index()

    list_of_table_retrievers = [
        (
            f"Simple search using {get_config().huggingface_embed_model}",
            heavydb_index.simple_search_for_table_names,
            lambda x: set(x),
        )
    ]

    for desc, func, res_parser in list_of_table_retrievers:
        print(desc)
        (
            total_overshot_correct,
            total_exactly_correct,
            total_partially_correct,
            total_incorrect,
        ) = await test_table_retrieval(question_dicts, func, res_parser)
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Path to sample.csv file required.")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
