import re

from llama_index.core.node_parser import MetadataAwareTextSplitter


class TableSchemaSplitter(MetadataAwareTextSplitter):
    """
    Helps to split HeavyDB table schema into chunks of text.
    """

    # compiled regex used to grab table and column comments from table create query response.
    comment_regex = re.compile(
        r"(?mi)create table (?P<table_name>\S+)\s*(?:/\*\s*(?P<table_comment>.*?)\s*\*\/)?\s*\($|^\s*(\w\S*).*?(?:\/\*\s*(.*?)\s*\*\/)?(?:,|\);)?$"
    )

    @classmethod
    def class_name(cls: type["TableSchemaSplitter"]) -> str:
        return "TableSchemaSplitter"

    def split_text_metadata_aware(self, text: str, metadata_str: str) -> list[str]:
        return self._split_text(text)

    def split_text(self, text: str) -> list[str]:
        return self._split_text(text)

    def get_table_comments(self, table_schema: str) -> tuple[str, str, list[tuple[str, str]]]:
        """
        Get table_name, table_comment, column_name, column_comment strings.
        """
        tname, tcomment, columns = "", "", []

        matches = self.comment_regex.findall(table_schema)
        for found in matches:
            table_name, table_comment, column, column_comment = found
            if table_name:
                tname = table_name
            if table_comment:
                tcomment = table_comment
            if column:
                columns.append((column, column_comment))

        return tname, tcomment, columns

    def _split_text(self, text: str) -> list[str]:
        """
        splits the table schema into chunks of text.
        """
        if text == "":
            return [text]

        splits = []
        table_name, table_comment, columns = self.get_table_comments(text)
        if table_comment:
            splits.append(f"{table_name}: {table_comment}")
        column_text_template = "{table_name}.{column_name}: {column_comment}"
        for col_name, col_comment in columns:
            splits.append(
                column_text_template.format(table_name=table_name, column_name=col_name, column_comment=col_comment)
            )

        return splits
