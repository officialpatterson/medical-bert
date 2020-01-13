# Currently provides two approaches for data loading.
from datareader.datareader import DataReader
from datareader.no_cls_and_sep_datareader import NoClsAndSepDataReader

class DataReaderFactory:
    def __init__(self, config):
        self._datareaders = {"one-doc": DataReader, "no-cls-sep": NoClsAndSepDataReader}
        self.config = config

    def register_datareader(self, name, datareader):
        self._datareaders[name] = datareader

    def make_datareader(self, name, tokenizer):
        datareader = self._datareaders.get(name)
        if not datareader:
            raise ValueError(format)
        return datareader(self.config, tokenizer)
