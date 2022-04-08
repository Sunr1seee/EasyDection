class BASE(dict):
    def __init__(self, *args, **kwargs):
        super(BASE, self).__init__(*args, **kwargs)
        self._dict = {}

    def register(self, target):
        def add_register_item(key, value):
            if not callable(value):
                raise Exception(f"CNN object must be callable! But receice:{value} is not callable!")
            if key in self._dict:
                print(f"warning: \033[33m{value.__name__} has been registered before, so we will overriden it\033[0m")
            self[key] = value
            return value

        if callable(target):  # 如果传入的目标可调用，说明之前没有给出注册名字，我们就以传入的函数或者类的名字作为注册名
            return add_register_item(target.__name__, target)
        else:  # 如果不可调用，说明额外说明了注册的可调用对象的名字
            return lambda x: add_register_item(target, x)

    def __call__(self, target):
        return self.register(target)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()


class CNN(BASE):
    def __init__(self, *args, **kwargs):
        super(CNN, self).__init__(*args, **kwargs)
        self._dict = {}


class ACTIVATE(BASE):
    def __init__(self, *args, **kwargs):
        super(ACTIVATE, self).__init__(*args, **kwargs)
        self._dict = {}

class CONTACT(BASE):
    def __init__(self, *args, **kwargs):
        super(CONTACT, self).__init__(*args, **kwargs)
        self._dict = {}


class BLOCK(BASE):
    def __init__(self, *args, **kwargs):
        super(BLOCK, self).__init__(*args, **kwargs)
        self._dict = {}


class DETECT(BASE):
    def __init__(self, *args, **kwargs):
        super(DETECT, self).__init__(*args, **kwargs)
        self._dict = {}


class LOSS(BASE):
    def __init__(self, *args, **kwargs):
        super(LOSS, self).__init__(*args, **kwargs)
        self._dict = {}


base_register = BASE()
cnn_register = CNN()
activate_register = ACTIVATE()
contact_register = CONTACT()
block_register = BLOCK()
detect_register = DETECT()
loss_register = LOSS()
