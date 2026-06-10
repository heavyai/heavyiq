# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from heavyiq.langchain.utils import retrieve_schema_details

column_details = """CREATE TABLE mlb_pitches /* mlb_pitches comment with 
newline chars */ (
SHAPE_LEN DOUBLE
BASE_BBL TEXT (4163500400, 4163500300, 4163400050 ...)
MPLUTO_BBL TEXT (4163500400, 4163500300, 4163400050 ...),
GEOMSOURCE TEXT (Photogramm, Other (Man, Other)
GLOBALID TEXT ({31298F86-3088-4F53-B3DB-71A9EFA6FA1F}, {F5F8CDA5-69E2-46F8-8F69-BA95C025B520}, {9F644794-F72C-4582-9E5E-B337E2B97068} ...)
geom MULTIPOLYGON
construction_date DATE (1652-01-01, 2023-01-01)
FID INTEGER,
STATE_NAME TEXT ENCODING DICT(32) (Twitter for iPhone, Twitter for Android, Instagram ...) /* foo */,
  SQMI DOUBLE,
  GlobalID TEXT ENCODING DICT(32),
  SHAPE_Leng DOUBLE[],
  SHAPE_Area DOUBLE[2],
  geom GEOMETRY(MULTIPOLYGON, 4326) ENCODING COMPRESSED(32)
  geom_new GEOMETRY(MULTIPOLYGON, 4326) ENCODING COMPRESSED(32) (sadds,sda,sda,sdsd) /* bqda
  multiline comment
  new
  sdnbds*/
"a var with space" TEXT ENCODING COMPRESSED(32) (sadds,sda,sda,sdsd) /* bqda sdnbds*/);
"""
expected_output = [
    ("SHAPE_LEN", "DOUBLE", None),
    ("BASE_BBL", "TEXT", None),
    ("MPLUTO_BBL", "TEXT", None),
    ("GEOMSOURCE", "TEXT", None),
    ("GLOBALID", "TEXT", None),
    ("geom", "MULTIPOLYGON", None),
    ("construction_date", "DATE", None),
    ("FID", "INTEGER", None),
    ("STATE_NAME", "TEXT ENCODING DICT(32)", "foo"),
    ("SQMI", "DOUBLE", None),
    ("GlobalID", "TEXT ENCODING DICT(32)", None),
    ("SHAPE_Leng", "DOUBLE[]", None),
    ("SHAPE_Area", "DOUBLE[2]", None),
    ("geom", "GEOMETRY(MULTIPOLYGON, 4326) ENCODING COMPRESSED(32)", None),
    ("geom_new", "GEOMETRY(MULTIPOLYGON, 4326) ENCODING COMPRESSED(32)", "bqda\n  multiline comment\n  new\n  sdnbds"),
    ('"a var with space"', "TEXT ENCODING COMPRESSED(32)", "bqda sdnbds"),
]


def test_schema():
    """
    From the cached table schema, parse and return parts for each column in list of tuples patterm.
    """
    details = retrieve_schema_details(column_details.strip())
    assert details["name"] == "mlb_pitches" and details["comment"] == "mlb_pitches comment with \nnewline chars"
    for parts, expected in zip(details["columns"], expected_output):
        assert parts == expected
