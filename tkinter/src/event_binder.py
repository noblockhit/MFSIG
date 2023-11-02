

global registered_binds
registered_binds = {} ## key: [array of functions]


class ParentFunction:
    def __init__(self, event_type):
        self.event_type = event_type
        self.functions = []
        self.bind(event_type, self)
    
    def __call__(self, *args, **kwargs):
        for f in self.functions:
            f(*args, **kwargs)

    def add_function(self, func):
        self.functions.append(func)


def register_bind(event_type, function):
    global registered_binds

    if event_type not in registered_binds.keys():
        registered_binds[event_type] = ParentFunction(event_type)

    registered_binds[event_type].add_function(function)


    
