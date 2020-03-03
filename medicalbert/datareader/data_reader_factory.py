# Currently provides two approaches for data loading.
from datareader.chunked_data_reader import ChunkedDataReader
from datareader.StandardDataReader import StandardDataReader


class DataReaderFactory:
    def __init__(self, config):
        self._datareaders = {"one-doc": StandardDataReader, "chunked": ChunkedDataReader}
        self.config = config

    def register_datareader(self, name, datareader):
        self._datareaders[name] = datareader

    def make_datareader(self, name, tokenizer):
        datareader = self._datareaders.get(name)
        if not datareader:
            raise ValueError(format)
        return datareader(self.config, tokenizer)
