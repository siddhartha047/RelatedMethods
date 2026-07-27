class DotMap(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        data = dict(*args, **kwargs)
        for key, value in data.items():
            self[key] = value

    @staticmethod
    def _wrap(value):
        if isinstance(value, dict) and not isinstance(value, DotMap):
            return DotMap(value)
        if isinstance(value, list):
            return [DotMap._wrap(item) for item in value]
        return value

    def __getattr__(self, item):
        return self.get(item, None)

    def __setattr__(self, key, value):
        self[key] = value

    def __setitem__(self, key, value):
        super().__setitem__(key, self._wrap(value))

    def toDict(self):
        result = {}
        for key, value in self.items():
            if isinstance(value, DotMap):
                result[key] = value.toDict()
            elif isinstance(value, list):
                result[key] = [
                    item.toDict() if isinstance(item, DotMap) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result
