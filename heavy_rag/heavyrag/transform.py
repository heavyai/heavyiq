import re
from collections.abc import Sequence
from typing import Any

from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.node_parser import MetadataAwareTextSplitter
from llama_index.core.schema import BaseNode, Document, MetadataMode, NodeRelationship


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
        if table_name:
            splits.append(f"{table_name}: {table_comment}")
        column_text_template = "{table_name}.{column_name}: {column_comment}"
        for col_name, col_comment in columns:
            splits.append(
                column_text_template.format(
                    table_name=table_name,
                    column_name=col_name,
                    column_comment=col_comment,
                )
            )

        return splits

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[BaseNode]:
        """Parse documents into nodes.
        Overrided method which adds extra metadata to the child nodes.
        Args:
            documents (Sequence[Document]): documents to parse
            show_progress (bool): whether to show progress bar

        """
        doc_id_to_document = {doc.id_: doc for doc in documents}

        with self.callback_manager.event(
            CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        ) as event:
            nodes = self._parse_nodes(documents, show_progress=show_progress, **kwargs)

            for i, node in enumerate(nodes):
                if node.ref_doc_id is not None and node.ref_doc_id in doc_id_to_document:
                    ref_doc = doc_id_to_document[node.ref_doc_id]
                    start_char_idx = ref_doc.text.find(node.get_content(metadata_mode=MetadataMode.NONE))

                    # update start/end char idx
                    if start_char_idx >= 0:
                        node.start_char_idx = start_char_idx
                        node.end_char_idx = start_char_idx + len(node.get_content(metadata_mode=MetadataMode.NONE))

                    # update metadata
                    if self.include_metadata:
                        # set node type metadata based on it's content
                        parent_metadata = doc_id_to_document[node.ref_doc_id].metadata
                        node_type = "table"
                        out = node.get_content().split(":", 1)
                        if out and "." in out[0]:
                            # column node
                            node_type = "column"
                        metadata = {**parent_metadata, "type": node_type}
                        node.metadata.update(metadata)

                if self.include_prev_next_rel:
                    if i > 0:
                        node.relationships[NodeRelationship.PREVIOUS] = nodes[i - 1].as_related_node_info()
                    if i < len(nodes) - 1:
                        node.relationships[NodeRelationship.NEXT] = nodes[i + 1].as_related_node_info()

            event.on_end({EventPayload.NODES: nodes})

        return nodes
