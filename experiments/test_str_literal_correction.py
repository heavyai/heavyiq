import argparse
import heavyai
import re
from copy import deepcopy

from typing import TypedDict, List, Tuple

class StringLiteralOp(TypedDict):
    """
    A type representation for string literal operations.
    """

    operator: str
    literal: str
    database: str
    table: str
    column: str

def correct_string_literal(con, literal: StringLiteralOp, exact_match_threshold):

    print(f"Correcting string literal: {literal['column']} : {literal['literal']}")
    case_match_query = f"SELECT {literal['column']}, COUNT(*) FROM {literal['database']}.{literal['table']} WHERE {literal['column']} ILIKE '{literal['literal']}' GROUP BY {literal['column']} ORDER BY COUNT(*) DESC;"
    cursor = con.execute(case_match_query)
    case_match_rows = cursor.fetchall()

    num_case_match_rows = len(case_match_rows)
    exact_match_count = 0
    total_count = 0
    altered_literal = deepcopy(literal)

    for row in case_match_rows:
        if row[0] == literal["literal"]:
            exact_match_count += row[1]  # type: ignore
        total_count += row[1]  # type: ignore

    if total_count > 0 and literal["operator"] != "ILIKE":
        if exact_match_count == 0 and num_case_match_rows == 1:
            altered_literal["literal"] = str(case_match_rows[0][0])
            return literal
        elif exact_match_count / total_count < exact_match_threshold:
            if literal["operator"] == "<>":
                altered_literal["operator"] = "NOT ILIKE"
            else:
                altered_literal["operator"] = "ILIKE"
            return altered_literal
    elif total_count == 0:
        lower_literal = literal["literal"].lower()
        lower_literal_prefix = re.split(r'[ ,:]+', lower_literal)[0]
        using_lower_literal_prefix = False if lower_literal_prefix == lower_literal else True
        prefix_condition = f"lower_attr ILIKE '{lower_literal_prefix}' OR lower_attr ILIKE '{lower_literal_prefix} %'" if using_lower_literal_prefix else f"lower_attr ILIKE '{lower_literal_prefix}%'"

        prefix_query = f"WITH distinct_values AS (SELECT LOWER({literal['column']}) AS lower_attr, COUNT(*) AS num_str_values FROM {literal['database']}.{literal['table']} GROUP BY LOWER({literal['column']})) SELECT lower_attr, num_str_values FROM distinct_values WHERE {prefix_condition} ORDER BY num_str_values DESC LIMIT 10;"
        cursor = con.execute(prefix_query)
        prefix_matches = cursor.fetchall()

        num_prefix_matches = len(prefix_matches)
        print(f"Num prefix matches: {num_prefix_matches}")
        print(f"Prefix matches: {prefix_matches}")

        similarity_query = f"WITH distinct_values AS (SELECT LOWER({literal['column']}) AS lower_attr, COUNT(*) AS num_str_values FROM {literal['database']}.{literal['table']} GROUP BY LOWER({literal['column']})) SELECT lower_attr, LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') - ABS(LENGTH(lower_attr) - LENGTH('{lower_literal}')) AS subset_distance, LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') AS absolute_distance, ABS(LENGTH(lower_attr) - LENGTH('{lower_literal}')) AS abs_length_difference, num_str_values FROM distinct_values WHERE LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') - ABS(LENGTH(lower_attr) - LENGTH('{lower_literal}')) < 5 AND CAST(LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') AS DOUBLE) / NULLIF(LENGTH('{lower_literal}'), 0) < 0.3 ORDER BY subset_distance ASC, abs_length_difference ASC, num_str_values DESC LIMIT 2;"

        cursor = con.execute(similarity_query)
        similarity_matches = cursor.fetchall()

        num_similarity_matches = len(similarity_matches)

        print(f"Num similarity matches: {num_similarity_matches}")
        print(f"Similarity matches: {similarity_matches}")

        if num_prefix_matches > 0 and num_prefix_matches <= 5:
            prefix_set = set() 
            for prefix_match in prefix_matches:
                prefix_set.add(prefix_match[0])
            num_similarity_prefix_overlaps = 0
            for similarity_match in similarity_matches:
                if similarity_match[0] in prefix_set:
                    num_similarity_prefix_overlaps += 1
            print(f"Num Similarity Prefix overlaps: {num_similarity_prefix_overlaps}")
            if num_similarity_prefix_overlaps != 1:
                if num_prefix_matches > 1:
                    # Multiple prefix matches, use string prefix
                    altered_literal["literal"] = f"{lower_literal_prefix}%"
                else:
                    # Only one prefix match, use the full matched string
                    altered_literal["literal"] = prefix_matches[0][0]
                if literal["operator"] in ("<>", "!=", "NOT LIKE", "NOT PG_ILIKE", "NOT ILIKE", "NOT PG_ILIKE"):
                    altered_literal["operator"] = "NOT ILIKE"
                else:
                    altered_literal["operator"] = "ILIKE"
                return altered_literal

        # If we are here, we do fuzzy similarity search

        if num_similarity_matches > 0:
            if (
                num_similarity_matches > 1
                and similarity_matches[0][1] == 0
                and similarity_matches[0][1] == similarity_matches[1][1]
            ):
                # Here there are at least two matches such that the user-provided literal is a full substring of
                # the column value. In this case, we will match against all strings that our string literal
                # is a substring of
                altered_literal["literal"] = f"%{lower_literal}%"
            else:
                # There was only one match, or two matches, so pick the
                # top returned value. In the case of a tie, pick the prefix match if it exists (we've sorted in ascending order by score and descending order by number
                if num_similarity_matches > 1 and similarity_matches[0][1] == similarity_matches[1][1]:
                    def matching_prefix_length(s1, s2):
                        match_length = 0
                        for c1, c2 in zip(s1, s2):
                            if c1 == c2:
                                match_length += 1
                            else:
                                break
                        return match_length
                    prefix_match_len_1 = matching_prefix_length(lower_literal, similarity_matches[0][0]) 
                    prefix_match_len_2 = matching_prefix_length(lower_literal, similarity_matches[1][0]) 
                    if prefix_match_len_1 >= prefix_match_len_2:
                        altered_literal["literal"] = str(similarity_matches[0][0])
                    else:
                        altered_literal["literal"] = str(similarity_matches[1][0])
                else:
                    altered_literal["literal"] = str(similarity_matches[0][0])
        if literal["operator"] in ("<>", "!=", "NOT LIKE", "NOT PG_ILIKE", "NOT ILIKE", "NOT PG_ILIKE"):
            altered_literal["operator"] = "NOT ILIKE"
        else:
            altered_literal["operator"] = "ILIKE"
        return altered_literal

    return altered_literal

def main():
    parser = argparse.ArgumentParser(description="Test various cases of StringLiteralOp corrections.")
    parser.add_argument("--host", help="Database host", default="localhost")
    parser.add_argument("--user", help="Database user", default="admin")
    parser.add_argument("--password", help="Database password", default="HyperInteractive")
    #parser.add_argument("--database", required=True, help="Database name")
    #parser.add_argument("--literal", required=True, help="Literal to be corrected")
    #parser.add_argument("--column", required=True, help="Column to search in")
    #parser.add_argument("--table", required=True, help="Table to search in")
    #parser.add_argument("--operator", required=True, help="Operator to use")
    parser.add_argument("--threshold", type=float, default=0.5, help="Exact match threshold")
    
    args = parser.parse_args()

    con = heavyai.connect(host=args.host, user=args.user, password=args.password, dbname="heavyai") 

    # Instantiating a StringLiteralOp
    string_literal_op: StringLiteralOp = {
        "operator": "=",
        "literal": "Alex Badn",
        "database": "slack", 
        "table": "slack",
        "column": "sender"
    }

    test_cases: List[Tuple[StringLiteralOp, StringLiteralOp]] = [
        (
            {"operator": "=", "literal": "Alex Badn", "database": "slack", "table": "slack", "column": "sender"},
            {"operator": "ILIKE", "literal": "alex baden", "database": "slack", "table": "slack", "column": "sender"},
        ),
        (
            {"operator": "=", "literal": "Alex Suhan", "database": "slack", "table": "slack", "column": "sender"},
            {"operator": "ILIKE", "literal": "alex Şuhan", "database": "slack", "table": "slack", "column": "sender"},
        ),
        (
            {"operator": "=", "literal": "Devon Energy", "database": "og_well_prod", "table": "oil_gas_wells", "column": "operator"},
            {"operator": "ILIKE", "literal": "devon%", "database": "og_well_prod", "table": "oil_gas_wells", "column": "operator"},
        ),
        (
            {"operator": "=", "literal": "Exxon", "database": "og_well_prod", "table": "oil_gas_wells", "column": "operator"},
            {"operator": "ILIKE", "literal": "exxonmobil", "database": "og_well_prod", "table": "oil_gas_wells", "column": "operator"},
        ),
        (
            {"operator": "=", "literal": "conoco", "database": "og_well_prod", "table": "oil_gas_wells", "column": "operator"},
            {"operator": "ILIKE", "literal": "conocophillips", "database": "og_well_prod", "table": "oil_gas_wells", "column": "operator"},
        ),
        (
            {"operator": "=", "literal": "tx", "database": "ca_tweets", "table": "ca_tweets", "column": "state_abbr"},
            {"operator": "ILIKE", "literal": "tx", "database": "ca_tweets", "table": "ca_tweets", "column": "state_abbr"},
        ),
        (
            {"operator": "=", "literal": "United Airlines", "database": "flights_2008", "table": "flights_2008", "column": "carrier_name"},
            {"operator": "ILIKE", "literal": "united air lines", "database": "flights_2008", "table": "flights_2008", "column": "carrier_name"},
        ),
        (
            {"operator": "=", "literal": "Joseph Biden", "database": "us_elections", "table": "us_county_pres_results_2000_2020", "column": "candidate"},
            {"operator": "ILIKE", "literal": "joseph r biden jr", "database": "us_elections", "table": "us_county_pres_results_2000_2020", "column": "candidate"},
        ),
        (
            {"operator": "=", "literal": "Joe Biden", "database": "us_elections", "table": "us_county_pres_results_2000_2020", "column": "candidate"},
            {"operator": "ILIKE", "literal": "joseph r biden jr", "database": "us_elections", "table": "us_county_pres_results_2000_2020", "column": "candidate"},
        ),
        (
            {"operator": "=", "literal": "Donald J. Trump", "database": "us_elections", "table": "us_county_pres_results_2000_2020", "column": "candidate"},
            {"operator": "ILIKE", "literal": "donald%", "database": "us_elections", "table": "us_county_pres_results_2000_2020", "column": "candidate"},
        ),
        (
            {"operator": "=", "literal": "1833 Waldorf", "database": "florida_parcels_2020", "table": "florida_parcels_2020", "column": "PHYADDR1"},
            {"operator": "ILIKE", "literal": "1833 waldorf dr", "database": "florida_parcels_2020", "table": "florida_parcels_2020", "column": "PHYADDR1"},
        ),
        (
            {"operator": "=", "literal": "16672 76th trail", "database": "florida_parcels_2020", "table": "florida_parcels_2020", "column": "PHYADDR1"},
            {"operator": "ILIKE", "literal": "16672 76th trl n", "database": "florida_parcels_2020", "table": "florida_parcels_2020", "column": "PHYADDR1"},
        ),
        (
            {"operator": "=", "literal": "16772 76th trail", "database": "florida_parcels_2020", "table": "florida_parcels_2020", "column": "PHYADDR1"},
            {"operator": "ILIKE", "literal": "16672 76th trl n", "database": "florida_parcels_2020", "table": "florida_parcels_2020", "column": "PHYADDR1"},
        ),

    ]
    num_passes = 0
    for input_literal, expected_output_literal in test_cases:
      corrected_literal = correct_string_literal(con, input_literal, args.threshold)
      print("Corrected Literal:", corrected_literal)
      test_passed = corrected_literal == expected_output_literal
      print(f"PASS: {test_passed}")
      num_passes += int(test_passed)
    con.close()
    print(f"Tests passed: {num_passes}/{len(test_cases)}")
    

if __name__ == "__main__":
    main()
