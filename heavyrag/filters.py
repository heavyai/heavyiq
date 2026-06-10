# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Module contain different filters used for node retrieval

from llama_index.core.vector_stores import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters

table_filter = MetadataFilter(key="type", operator=FilterOperator.EQ, value="table")
document_filter = MetadataFilter(key="type", operator=FilterOperator.EQ, value="document")
facts_filter = MetadataFilter(key="type", operator=FilterOperator.EQ, value="facts")
table_filters = MetadataFilters(filters=[table_filter])
document_filters = MetadataFilters(filters=[document_filter])
facts_filters = MetadataFilters(filters=[facts_filter])


def get_table_filter_matches(heavydb_name: str, table_name: str | None = None) -> MetadataFilters:
    """
    Table filter which was extended to match a particular tablename.
    """
    filters = [table_filter, MetadataFilter(key="dbname", operator=FilterOperator.EQ, value=heavydb_name)]
    if table_name:
        filters.append(MetadataFilter(key="name", operator=FilterOperator.EQ, value=table_name))
    return MetadataFilters(
        filters=filters,  # type: ignore
        condition=FilterCondition.AND,
    )


def get_document_filter_matches(file_name: str) -> MetadataFilters:
    """
    Document filter which was extended to match a particular filename.
    This helps to filterout the nodes which are relevant to a particular filename.
    """
    return MetadataFilters(
        filters=[document_filter, MetadataFilter(key="name", operator=FilterOperator.EQ, value=file_name)],
        condition=FilterCondition.AND,
    )


def get_facts_filter_matches(heavydb_name: str) -> MetadataFilters:
    """
    Facts filter which was extended to match a particular heavydb database name.
    """
    return MetadataFilters(
        filters=[facts_filter, MetadataFilter(key="dbname", operator=FilterOperator.EQ, value=heavydb_name)],
        condition=FilterCondition.AND,
    )
