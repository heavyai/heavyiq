from modules.langchain.index import heavydb_index


def main():
    res = heavydb_index.ask_about_database(
        "I need the schemas for tables containing crop acreage and Mixed Pasture information for the year 2014"
    )
    print(res)
    return


if __name__ == "__main__":
    main()
