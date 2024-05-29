# Module contain different filters used for node retrieval

from llama_index.core.vector_stores import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters

table_filter = MetadataFilter(key="type", operator=FilterOperator.EQ, value="table")
document_filter = MetadataFilter(key="type", operator=FilterOperator.EQ, value="document")
table_filters = MetadataFilters(filters=[table_filter])
document_filters = MetadataFilters(filters=[document_filter])


def get_table_filter_matches(table_name: str) -> MetadataFilters:
    """
    Table filter which was extended to match a particular tablename.
    """
    return MetadataFilters(
        filters=[table_filter, MetadataFilter(key="name", operator=FilterOperator.EQ, value=table_name)],
        condition=FilterCondition.AND,
    )


def get_document_filter_matches(file_name: str) -> MetadataFilters:
    """
    Document filter which was extended to match a particular filename.
    This helps to filterout the nodes which are relevant to a particular
    """
    return MetadataFilters(
        filters=[document_filter, MetadataFilter(key="name", operator=FilterOperator.EQ, value=file_name)],
        condition=FilterCondition.AND,
    )
