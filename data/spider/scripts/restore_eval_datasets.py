import heavyai
import pathlib

eval_databases = [
    "concert_singer",
    "pets_1",
    "employee_hire_evaluation",
    "cre_Doc_Template_Mgt",
    "course_teach",
    "museum_visit",
    "battle_death",
    "student_transcripts_tracking",
    "tvshow",
    "poker_player",
    "voter_1",
    "world_1",
    "orchestra",
    "network_1",
    "dog_kennels",
    "singer",
    "real_estate_properties",
]

local_import_base_dir = "/home/paperspace/heavyai/heavyiq_eval"
heavyai_con = heavyai.connect(user="admin", password="HyperInteractive", host="localhost", dbname="heavyai")
for db in eval_databases:
    create_db_sql = f"CREATE DATABASE IF NOT EXISTS {db}"
    heavyai_con.execute(create_db_sql)

for db in eval_databases:
    print(f"Database: {db}")
    con = heavyai.connect(user="admin", password="HyperInteractive", host="localhost", dbname=db)
    local_import_dir = f"{local_import_base_dir}/{db}"
    dump_files = pathlib.Path(local_import_dir).glob("*.dump.lz4")
    print(dump_files)
    tables = [str(dump_file).split(".")[0].split("/")[-1] for dump_file in dump_files]
    for table in tables:
        print(f"Table: {table}")
        restore_sql = f"RESTORE TABLE {table} FROM '{local_import_dir}/{table}.dump.lz4' WITH (COMPRESSION = 'LZ4');"
        print(restore_sql)
        try:
            con.execute(restore_sql)
        except Exception as e:
            print(e)
