# Lazy imports to avoid initializing chains during module load
# This prevents SIGSEGV issues with --preload when using ChromaDB/RAG
from .heavydb.question_chain import chain as question_chain
from .heavydb.sql_chain import chain as sql_chain
from .heavydb.sql_cot_chain import chain as sql_cot_chain
from .heavydb.sql_gen_chain import chain as sql_gen_chain
from .heavydb.table_chain import chain as table_chain
from .heavydb.table_chain import tables_list_chain
from .heavydb.answer_chain import chain as answer_chain
from .heavydb.auto_answer_chain import chain as auto_answer_chain
from .heavydb.auto_sql_chain import chain as auto_sql_chain

# _chains_cache = {}

# def _get_chain(chain_name: str):
#     """Lazy load chains on first access."""
#     if chain_name not in _chains_cache:
#         if chain_name == 'answer_chain':
#             from .heavydb.answer_chain import chain
#             _chains_cache[chain_name] = chain
#         elif chain_name == 'auto_answer_chain':
#             from .heavydb.auto_answer_chain import chain
#             _chains_cache[chain_name] = chain
#         elif chain_name == 'auto_sql_chain':
#             from .heavydb.auto_sql_chain import chain
#             _chains_cache[chain_name] = chain
#         elif chain_name == 'question_chain':
#             from .heavydb.question_chain import chain
#             _chains_cache[chain_name] = chain
#         elif chain_name == 'sql_chain':
#             from .heavydb.sql_chain import chain
#             _chains_cache[chain_name] = chain
#         elif chain_name == 'sql_cot_chain':
#             from .heavydb.sql_cot_chain import chain
#             _chains_cache[chain_name] = chain
#         elif chain_name == 'sql_gen_chain':
#             from .heavydb.sql_gen_chain import chain
#             _chains_cache[chain_name] = chain
#         elif chain_name == 'table_chain':
#             from .heavydb.table_chain import chain
#             _chains_cache[chain_name] = chain
#         elif chain_name == 'tables_list_chain':
#             from .heavydb.table_chain import tables_list_chain
#             _chains_cache[chain_name] = tables_list_chain
#     return _chains_cache[chain_name]

# # Export lazy-loaded chains using __getattr__
# def __getattr__(name: str):
#     """Lazy load chains when accessed as module attributes."""
#     if name in (
#         'answer_chain', 'auto_answer_chain', 'auto_sql_chain',
#         'question_chain', 'sql_chain', 'sql_cot_chain',
#         'sql_gen_chain', 'table_chain', 'tables_list_chain'
#     ):
#         return _get_chain(name)
#     raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# def __dir__():
#     return [
#         'answer_chain', 'auto_answer_chain', 'auto_sql_chain',
#         'question_chain', 'sql_chain', 'sql_cot_chain',
#         'sql_gen_chain', 'table_chain', 'tables_list_chain'
#     ]
