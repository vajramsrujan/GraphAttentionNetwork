import yaml


# Credit
# https://jonnyjxn.medium.com/how-to-config-your-machine-learning-experiments-without-the-headaches-bb379de1b957
class Configuration(object):
    """
    Implements lash-based retrieval of
    nested elements in configuration dictionary
    """

    def __init__(self, config_path):
        with open(config_path) as cf_file:
            cfg = yaml.safe_load(cf_file.read())

        self._data = cfg

    def get(self, path=None, default=None):
        # we need to deep-copy self._data to avoid over-writing its data
        sub_dict = dict(self._data)

        if path is None:
            return sub_dict

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        try:
            for path_item in path_items:
                sub_dict = sub_dict.get(path_item)

            value = sub_dict.get(data_item, default)

            return value
        except (TypeError, AttributeError):
            return default
