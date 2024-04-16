from functools import wraps
from typing import Callable

import pytest

from heavyiq.cli.eval_commands.utils import SQL


def eq(func: Callable[[], tuple[str, str]]):
    @wraps(func)
    def wrapper(*args, **kwargs) -> None:
        sqla, sqlb = func(*args, **kwargs)
        assert SQL(sqla) == SQL(sqlb)

    return wrapper


def neq(func: Callable[[], tuple[str, str]]):
    @wraps(func)
    def wrapper(*args, **kwargs) -> None:
        sqla, sqlb = func(*args, **kwargs)
        assert SQL(sqla) != SQL(sqlb)

    return wrapper


@eq
def test_sql_compare_equality_check_should_pass_for_queries_having_whitespaces() -> tuple[str, str]:
    return "select count(*) as num_count from foo;", "select count(* ) as num_count from  foo;"


@eq
def test_sql_compare_equlaity_check_should_pass_for_queries_having_case_incensitive_keywords() -> tuple[str, str]:
    return "select count(*) as num_count from foo;", "SELECT count(*) AS num_count from  foo;"


@eq
def test_sql_compare_equality_check_should_pass_for_queries_having_different_aliases() -> tuple[str, str]:
    return "select count(*) as num_count from foo;", "select count(* ) as no_count from  foo;"


@eq
def test_sql_compare_equality_check_should_pass_for_sql_queries_having_whietspaces_and_different_aliases() -> (
    tuple[str, str]
):
    sql1 = """SELECT T3.FIPS, CAST(COUNT(*) AS DOUBLE) / NULLIF(AVG(T3.POP2012), 0) AS num_starting_trips_per_capita, AVG(T3.MED_AGE) AS median_age FROM austin_bikeshare_trips AS T1 JOIN austin_bikeshare_stations AS T2 ON T1.start_station_name = T2.name JOIN austin_census_block_groups AS T3 ON ST_CONTAINS( T3.geom, ST_SETSRID(ST_POINT(T2.longitude, T2.latitude), 4326) ) WHERE DATE_TRUNC(YEAR, T1.start_time) = '2015-01-01 00:00:00' GROUP BY T3.FIPS;"""
    sql2 = """SELECT T3.FIPS, CAST(COUNT(*) AS DOUBLE) / NULLIF(AVG(T3.POP2012), 0) AS num_trips_per_capita, AVG(T3.MED_AGE) AS median_age FROM austin_bikeshare_trips AS T1 JOIN austin_bikeshare_stations AS T2 ON T1.start_station_name = T2.name JOIN austin_census_block_groups AS T3 ON ST_CONTAINS(T3.geom, ST_SETSRID(ST_POINT(T2.longitude, T2.latitude), 4326)) WHERE DATE_TRUNC(YEAR, T1.start_time) = '2015-01-01 00:00:00' GROUP BY T3.FIPS;"""
    return sql1, sql2


@neq
def test_sql_compare_equality_check_should_fail_for_sql_queries_having_differnet_identifies() -> tuple[str, str]:
    return "select * from foo where bar='sad'", "select * from foo where bar='ads'"


@neq
def test_sql_compare_equality_check_should_fail_for_queries_having_extra_characters() -> tuple[str, str]:
    return "select count(*) as num_count from foo where location='Texas';", "select count(* ) as num_count from  foo;"
