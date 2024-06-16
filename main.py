from injector import Injector
from modules import ProcessorModule
from processor import Processor


def main():
    file = '../data/tar/market.tar'
    injector = Injector([ProcessorModule])
    processor = injector.get(Processor)
    processor.process(file)


if __name__ == "__main__":
    main()
