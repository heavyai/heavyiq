import sys
import sqlite3

# Check the version of the native sqlite3 module
native_version = sqlite3.sqlite_version

# If the native version is less than 3.35, override with pysqlite3
if tuple(map(int, native_version.split('.'))) < (3, 35):
    import pysqlite3 as sqlite3
    sys.modules['sqlite3'] = sqlite3
