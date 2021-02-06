from torch import Tensor, nn

model_dict = {}


def register_model(cls):
    if cls.__name__ not in model_dict:
        model_dict[cls.__name__] = cls
    elif model_dict[cls.__name__] != cls:
        raise KeyError(f"Duplicated key {cls.__name__} from {model_dict[cls.__name__]} and {cls}!!!!")


def unpack_feature(x):
    if isinstance(x, tuple):
        f_list, x = x
    else:
        f_list = []
    return f_list, x


def pack_feature(f_list: list, x: Tensor, with_feature: bool = True, add: bool = True):
    if with_feature:
        if add:
            return f_list + [x], x
        else:
            return f_list, x
    else:
        return x


def append_with_feature(func):
    def wrapper(self, input, *inp, with_feature=False, **kwargs):
        f_list, x = func(self, input, *inp, **kwargs)
        return pack_feature(f_list, x, with_feature=with_feature, add=True)

    return wrapper


def record_feature(func):
    def wrapper(self, input, *inp, with_feature=True, **kwargs):
        f_list, x = unpack_feature(input)
        x = func(self, x, *inp, **kwargs)
        return pack_feature(f_list, x, with_feature=with_feature, add=True)

    return wrapper


def obj_record_feature(obj: nn.Module, default=True, record=True):
    def warpper(self, input, *inp, with_feature=default, **kwargs):
        f_list, x = unpack_feature(input)
        x = self.forward_(x, *inp, **kwargs)
        return pack_feature(f_list, x, with_feature=with_feature, add=record)

    from types import MethodType
    obj.forward_ = obj.forward
    obj.forward = MethodType(warpper, obj)
    return obj
