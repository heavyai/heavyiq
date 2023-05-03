from heavynl.langchain.index import get_heavydb_index


def main():
    heavydb_index = get_heavydb_index()
    # heavydb_index.ask
    # heavydb_index.ask_about_database
    # heavydb_index.ask_using_rephrased_question
    # heavydb_index.rephrase_question
    res = heavydb_index.ask_using_simple_search(
        "I need the schemas for tables containing crop acreage and Mixed Pasture information for the year 2014"
    )
    print(res)
    return


if __name__ == "__main__":
    main()
