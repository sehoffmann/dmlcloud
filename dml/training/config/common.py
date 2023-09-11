class ConfigVar:
    def _default_parse_fn(self, config, args):
        val = getattr(args, self.name, None)
        if val is not None:
            setattr(config, self.name, val)

    def __init__(self, add_argument_fn=None, parse_argument_fn=_default_parse_fn):
        if parse_argument_fn is ConfigVar._default_parse_fn:
            parse_argument_fn = self._default_parse_fn  # bind self
        self.add_argument_fn = add_argument_fn
        self.parse_argument_fn = parse_argument_fn

    def __set_name__(self, owner, name):
        self.name = name
        if not hasattr(owner, '_config_vars'):
            owner._config_vars = []
        owner._config_vars.append(self)

    def __get__(self, obj, objtype=None):
        return obj.dct[self.name]

    def __set__(self, obj, value):
        obj.dct[self.name] = value

    def __delete__(self, obj):
        del obj.dct[self.name]

    def add_argument(self, config, parser):
        if self.add_argument_fn:
            self.add_argument_fn(config, parser)

    def parse_argument(self, config, args):
        if self.parse_argument_fn:
            self.parse_argument_fn(config, args)


class ArgparseVar(ConfigVar):
    def _default_add_fn(self, config, parser):
        option = f'--{self.name.replace("_", "-")}'
        args = self.args or [option]
        kwargs = self.kwargs.copy()
        kwargs['dest'] = self.name
        parser.add_argument(*args, **kwargs)

    def __init__(self, *args, add_argument_fn=_default_add_fn, parse_argument_fn=ConfigVar._default_parse_fn, **kwargs):
        if add_argument_fn is ArgparseVar._default_add_fn:
            add_argument_fn = self._default_add_fn  # bind self

        super().__init__(add_argument_fn, parse_argument_fn)
        self.args = args
        self.kwargs = kwargs
        if 'dest' in self.kwargs:
            raise ValueError('dest cannot be specified in kwargs')


class SubConfig:
    def __init__(self, parent, root_dct, key=None):
        self.parent = parent
        self.root_dct = root_dct
        self.key = key
        self.set_defaults()

    def set_defaults(self):
        pass

    @property
    def dct(self):
        if self.key is None:
            return self.root_dct
        else:
            return self.root_dct[self.key]

    def add_arguments(self, parser):
        for cfg_var in self._config_vars:
            cfg_var.add_argument(self, parser)

    def parse_args(self, args):
        for cfg_var in self._config_vars:
            cfg_var.parse_argument(self, args)
