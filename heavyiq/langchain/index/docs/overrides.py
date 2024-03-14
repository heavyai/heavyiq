from typing import Optional
from urllib.parse import urljoin

from langchain_community.document_loaders import GitbookLoader
from langchain_core.documents import Document


class OverrideGitbookLoader(GitbookLoader):
    def __init__(
        self,
        web_page: str,
        load_all_paths: bool = False,
        base_url: Optional[str] = None,
        content_selector: str = "main",
        continue_on_failure: Optional[bool] = False,
        exclude_paths: list[str] = [],
        include_paths: list[str] = [],
    ):
        """Initialize with web page and whether to load all paths.

        Args:
            web_page: The web page to load or the starting point from where
                relative paths are discovered.
            load_all_paths: If set to True, all relative paths in the navbar
                are loaded instead of only `web_page`.
            base_url: If `load_all_paths` is True, the relative paths are
                appended to this base url. Defaults to `web_page`.
            content_selector: The CSS selector for the content to load.
                Defaults to "main".
            continue_on_failure: whether to continue loading the sitemap if an error
                occurs loading a url, emitting a warning instead of raising an
                exception. Setting this to True makes the loader more robust, but also
                may result in missing data. Default: False
        """
        super().__init__(web_page, load_all_paths, base_url, content_selector, continue_on_failure)
        self.exclude_paths = exclude_paths
        self.include_paths = include_paths

    def load(self) -> list[Document]:
        """Fetch text from one single GitBook page."""
        if self.load_all_paths:
            soup_info = self.scrape()
            relative_paths = self._get_paths(soup_info)
            relative_paths = [path for path in relative_paths if not any(ip in path for ip in self.exclude_paths)]
            relative_paths = [path for path in relative_paths if any(ip in path for ip in self.include_paths)]
            urls = [urljoin(self.web_path, path) for path in relative_paths]
            soup_infos = self.scrape_all(urls)
            _documents = [self._get_document(soup_info, url) for soup_info, url in zip(soup_infos, urls)]
        else:
            soup_info = self.scrape()
            _documents = [self._get_document(soup_info, self.web_path)]
        documents = [d for d in _documents if d]

        return documents
