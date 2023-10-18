import asyncio
from heavyiq.langchain.index import aget_heavydb_index


async def main():
    heavydb_index = await aget_heavydb_index()
    # heavydb_index.ask
    # heavydb_index.ask_about_database
    # heavydb_index.ask_using_rephrased_question
    # heavydb_index.rephrase_question
    res = await heavydb_index.ask_about_database(
        "I need the schemas for tables containing crop acreage and Mixed Pasture information for the year 2014"
    )
    print(res)
    return


if __name__ == "__main__":
    asyncio.run(main())
