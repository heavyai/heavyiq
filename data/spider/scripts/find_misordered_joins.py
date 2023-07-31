import argparse
import json
import sys


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Find misordered join tables")
    parser.add_argument("-q", "--queries", help="Queries JSON file", default="./sql_all_prompts_v16.json")
    return parser.parse_args(argv)


def getQueries(queries_filename):
    data = None
    with open(queries_filename, "r") as f:
        data = json.load(f)
    queries = []
    for entry in data:
        db_id = entry["db_id"]
        query_id = entry["query_id"]
        query = entry["output"]
        queries.append({"db_id": db_id, "query_id": query_id, "query": query})
    return queries


def isJoinMisordered(query):
    max_num_joins = 8
    join_alias_idxs = []
    for join_idx in range(max_num_joins):
        join_alias = f" AS T{join_idx}"
        join_alias_idxs.append(query.find(join_alias))
    join_misordered = False
    for join_alias_idx in range(1, len(join_alias_idxs)):
        if join_alias_idxs[join_alias_idx] == -1:
            break
        if join_alias_idxs[join_alias_idx] < join_alias_idxs[join_alias_idx - 1]:
            join_misordered = True
            break
    return join_misordered


def doesJoinHaveNonCanonicalAliases(query):
    lower_query = query.lower()
    start_idx = 0
    join_aliases = []
    type_casts = {
        "tinyint",
        "smallint",
        "int",
        "bigint",
        "float",
        "double",
        "decimal",
        "numeric",
        "date",
        "timestamp",
        "text",
    }
    while True:
        start_idx = lower_query.find(" as ", start_idx)
        if start_idx == -1:
            break
        join_alias_idx = start_idx + 4
        join_alias = ""
        while True:
            if join_alias_idx >= len(lower_query):
                break
            if not lower_query[join_alias_idx].isalnum() and lower_query[join_alias_idx] != "_":
                break
            join_alias += lower_query[join_alias_idx]
            join_alias_idx += 1
        start_idx = join_alias_idx
        if join_alias in type_casts:
            continue
        join_aliases.append(join_alias)
    # if len(join_aliases) > 0:
    #    print(join_aliases)

    for join_alias in join_aliases:
        if join_alias[0] != "t":
            return True
        if not join_alias[1].isnumeric():
            return True
        return False


def main(argv):
    options = getOptions(argv)
    queries = getQueries(options.queries)
    num_misordered_joins = 0
    num_non_canonical_aliases = 0
    for query in queries:
        if doesJoinHaveNonCanonicalAliases(query["query"]):
            print(query["db_id"], query["query_id"], query["query"])
            num_non_canonical_aliases += 1

        # join_is_misordered = isJoinMisordered(query["query"])
        # if join_is_misordered:
        #    num_misordered_joins += 1
        #    print(query["db_id"], query["query_id"], query["query"])

    print(num_non_canonical_aliases)
    print(len(queries))


if __name__ == "__main__":
    main(sys.argv[1:])
