from datasets.download.download_manager import DownloadManager, is_relative_path


def patch_download_manager():
    ori_download = DownloadManager._download
    def _download(self, url_or_filename: str, *args, **kwargs) -> str:
        url_or_filename = str(url_or_filename)
        if is_relative_path(url_or_filename) and self._base_path.startswith('http'):
            self._base_path = '.'
        return ori_download(self, url_or_filename, *args, **kwargs)
    DownloadManager._download = _download

patch_download_manager()
