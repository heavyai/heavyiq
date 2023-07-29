import time
import unittest
import multiprocessing
from heavynl.utils import LRUCache


class TestLRUCache(unittest.TestCase):
    """
    Test Class for LRUCache.
    """

    @staticmethod
    def process_one(cache: LRUCache):
        """
        LRUCache operations on process 1.
        """
        # Cache some data
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Test get method
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") is None  # key4 doesn't exist, should return None

        # Test updating existing key
        cache.put("key2", "new_value2")
        assert cache.get("key2") == "new_value2"

        # Test eviction when capacity is reached
        cache.put("key4", "value4")  # This will cause key1 to be evicted
        assert cache.get("key1") is None

        # Test LRU behavior (key3 should be evicted as it was the least recently used)
        cache.put("key5", "value5")
        assert cache.get("key3") is None
        # here the cache contains key2, key4, key5

    @staticmethod
    def process_two(cache: LRUCache):
        """
        LRUCache operations on process 2.
        """
        # here the cache contains key2, key4, key5
        cache.get("key2")
        # which make key2 as recently used key
        cache.put("key6", "value6")
        # this will remove key4
        # now the cache contains key5, key2, key6

    def test_lru_shared_cache_sequentially(self):
        """
        Test shared lru cache sequently, ie. one process modifes the caches after another.
        Always wait for all the child processes to complete before exiting the main process.
        """

        cache = LRUCache(capacity=3)

        # Create two processes
        process1 = multiprocessing.Process(target=self.process_one, args=(cache,))
        process2 = multiprocessing.Process(target=self.process_two, args=(cache,))

        # Start the process 1
        process1.start()
        # wait for process 1 to finish
        process1.join()

        # start the process 2
        process2.start()
        # wait for process 1 to finish
        process2.join()

        self.assertEqual(list(cache.order), ["key5", "key2", "key6"])

    @staticmethod
    def process_x(cache: LRUCache):
        """
        Cache alter method.
        """
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        # introduce a sleep of half a second, so that process y will put the key3
        time.sleep(0.5)
        # this time key3 gets updated and the key3 appears to be the recently used key
        cache.put("key3", "newvalue3")

    @staticmethod
    def process_y(cache: LRUCache):
        """
        Cache alter method.
        """
        cache.put("key3", "value3")
        time.sleep(0.6)  # extr 0.1 sec wait in-order to get the process_x completed.
        # make  key1 as recently accessed
        cache.get("key1")
        cache.put("key4", "key4")  # here key2 got evicted

    def test_lru_cache_parallely(self):
        """
        Test shared lru cache parallely, ie. two processes tries to modify the cache at the same time.
        In this case, we don't need to explicitly put the lock, manager.Dict (ProxyDict) automatically handles it.
        """

        cache = LRUCache(capacity=3)

        # Create two processes
        processx = multiprocessing.Process(target=self.process_x, args=(cache,))
        processy = multiprocessing.Process(target=self.process_y, args=(cache,))
        processes = [processx, processy]
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        self.assertEqual(list(cache.order), ["key3", "key1", "key4"])
        self.assertEqual(cache.get("key3"), "newvalue3")


if __name__ == "__main__":
    unittest.main()
