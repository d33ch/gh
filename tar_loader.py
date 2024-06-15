from bz2 import BZ2File
import bz2
import os
import tarfile
from typing import Iterator
from base_loader import BaseLoader


class TarLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def load(self, file_path: str) -> Iterator[BZ2File]:
        try:
            if not os.path.isfile(file_path) and ext == '.tar':
                ext = os.path.splitext(file_path)[1]
                with tarfile.TarFile(file_path) as archive:
                    for file in archive:
                        yield bz2.open(archive.extractfile(file))
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except tarfile.TarError as e:
            print(f"Error opening tar file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return None
