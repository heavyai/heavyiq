# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import csv
from collections import defaultdict
from typing import Any


def read_csv_as_lists(file_path: str, header: bool = True) -> list[Any]:
    with open(file_path, mode="r", newline="", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        if header:
            # skip the header row
            next(csv_reader)
        rows = [row for row in csv_reader]
    return rows


def read_and_group_csv(file_path: str, group_by_column_idx: int, header: bool = True) -> dict[str, list]:
    grouped_data = defaultdict(list)
    with open(file_path, mode="r", newline="", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        if header:
            # Skip the header row
            next(csv_reader, None)
        for row in csv_reader:
            # Assuming the second column is the column to group by (index 1)
            key = row[group_by_column_idx]
            grouped_data[key].append(row)
    return grouped_data


def write_csv(file_path: str, headers: list, rows: list):
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        if headers:
            csv_writer.writerow(headers)
        csv_writer.writerows(rows)


def string_to_list(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return None
