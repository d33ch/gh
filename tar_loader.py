from bz2 import BZ2File
import bz2
import os
import tarfile
from base_loader import BaseLoader


class TarLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def load(self, file_path: str) -> BZ2File:
        try:
            ext = os.path.splitext(file_path)[1]
            if os.path.isfile(file_path) and ext == '.tar':
                with tarfile.TarFile(file_path) as archive:
                    for file in archive:
                        print(file.name)
                        extracted = archive.extractfile(file)
                        if extracted is not None:
                            yield bz2.open(extracted)

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except tarfile.TarError as e:
            print(f"Error opening tar file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return None
