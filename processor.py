from base_data_parser import BaseDataParser
from base_loader import BaseLoader
from injector import inject


class Processor():
    loader: BaseLoader
    parser: BaseDataParser

    @inject
    def __init__(self, loader: BaseLoader, parser: BaseDataParser):
        self.loader = loader
        self.parser = parser

    def process(self, file):
        data = self.loader.load(file)
        for d in data[:1]:
            self.parser.parse(d)