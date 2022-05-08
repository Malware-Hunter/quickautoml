import json


class FileParser:
  def __init__(self):
    pass

  @staticmethod
  def read_json(filepath: str) -> dict:
    file = open(filepath)
    return json.load(file)


class ConfigFileParser(FileParser):
  def __init__(self):
    super().__init__()

  def parse(self, filepath: str):
    pass


class AlgorithmFileParser(FileParser):
  pass
