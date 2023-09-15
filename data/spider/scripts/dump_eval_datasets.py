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

local_export_base_dir = "/Users/todd//data/heavyai_dumps/heavyiq/eval"
export_base_dir = "/media/psf/Home/data/heavyai_dumps/heavyiq/eval"
for db in eval_databases:
    print(f"Database: {db}")
    con = heavyai.connect(user="admin", password="HyperInteractive", host="10.211.55.4", dbname=db)
    tables = con.get_tables()
    local_export_db_path = f"{local_export_base_dir}/{db}"
    export_db_path = f"{export_base_dir}/{db}"
    pathlib.Path(local_export_db_path).mkdir(parents=True, exist_ok=True)
    for table in tables:
        print(f"Table: {table}")
        export_path = f"{export_db_path}/{table}.dump.lz4"
        dump_table_sql = f"DUMP TABLE {table} TO '{export_path}' WITH (COMPRESSION = 'LZ4');"
        print(dump_table_sql)
        con.execute(dump_table_sql)

    print(tables)
