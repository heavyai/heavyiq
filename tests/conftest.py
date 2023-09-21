import pytest
import time


@pytest.fixture(autouse=True, scope="session")
def print_total_time(request):
    start_time = time.time()

    yield  # This is where the test cases run

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTotal time taken for all test cases: {total_time:.2f} seconds")
