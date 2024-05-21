import argparse
import sys

import csv
from collections import defaultdict


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Summarize HeavyIQ eval results")
    parser.add_argument("-i", "--input", help="Input CSV", required=True)
    return parser.parse_args(argv)


def main(argv):
    options = getOptions(argv)

    # Initialize variables for global statistics
    total_queries = 0
    successful_queries = 0

    # Initialize variables for db_id statistics
    db_stats = defaultdict(lambda: {"total": 0, "success": 0})

    # Initialize variables for status types
    status_stats = defaultdict(int)

    # Read the CSV file
    with open(options.input, "r") as csvfile:
        csvreader = csv.reader(csvfile)

        # Skip header
        header = next(csvreader)

        for row in csvreader:
            query_id, db_id, gold_query, pred_query, success, status, error = row
            success = success == "True"  # Convert string to boolean

            # Update global stats
            total_queries += 1
            if success:
                successful_queries += 1

            # Update db_id stats
            db_stats[db_id]["total"] += 1
            if success:
                db_stats[db_id]["success"] += 1

            # Update status stats
            status_stats[status] += 1

    # Calculate and print global statistics
    global_success_rate = (successful_queries / total_queries) * 100
    print(f"Global Statistics:")
    print(f"Total queries: {total_queries}")
    print(f"Successful queries: {successful_queries}")
    print(f"Success rate: {global_success_rate:.2f}%")

    # Calculate and print per db_id statistics
    print("\nDB_ID Statistics:")
    for db_id, stats in db_stats.items():
        db_success_rate = (stats["success"] / stats["total"]) * 100
        print(f"For db_id {db_id}:")
        print(f"\tTotal queries: {stats['total']}")
        print(f"\tSuccessful queries: {stats['success']}")
        print(f"\tSuccess rate: {db_success_rate:.2f}%")

    # Calculate and print statistics by status type
    print("\nStatus Type Statistics:")
    for status, count in status_stats.items():
        status_percentage = (count / total_queries) * 100
        print(f"For status {status}:")
        print(f"\tTotal queries: {count}")
        print(f"\tPercentage: {status_percentage:.2f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
