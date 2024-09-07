from injector import Injector
from dtos.processor_config import ProcessorConfig
from modules import ProcessorModule
from processor import Processor


def main():
    file = "./data/tar/market.tar"
    config = ProcessorConfig([(600, 15), (300, 10), (120, 5), (60, 1)])
    injector = Injector([ProcessorModule])
    processor = injector.get(Processor)
    processor.process(file, config)


if __name__ == "__main__":
    main()
