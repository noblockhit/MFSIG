from dataclasses import dataclass
from typing import Any, ClassVar, get_type_hints, Union, _UnionGenericAlias
import types
import werkzeug.serving as serving
if __name__ == '__main__':
    from deps import bmscam
else:
    from .deps import bmscam
from pathlib import Path


class ABSType:
    pass


class Meta(type):
    def __setattr__(self, __name: str, __value: Any) -> None:
        hints = get_type_hints(self)
        
        class_hint = hints.get(__name)
        if not class_hint:
            return super().__setattr__(__name, __value)
        
        hint = class_hint.__dict__.get("__args__")[0]
        
        can_be_none = isinstance(hint, _UnionGenericAlias)

        if can_be_none:
            hint = hint.__dict__.get("__args__")[0]
            
        val_type = type(__value)

        if isinstance(hint, types.FunctionType) or isinstance(hint, types.LambdaType) or hasattr(hint, "__self__"):
            print(hint, val_type, __value)
        else:
            if not (isinstance(__value, hint) or (__value is None and can_be_none)):
                if not (ABSType in hint.__bases__ and val_type.__name__ == hint.__name__):
                    raise ValueError(f"The property {__name} only takes {hint}, got {val_type} <{__value}> instead.")
        return super().__setattr__(__name, __value)

abs_motor_type = type("Motor", (ABSType,), dict())

@dataclass
class State(metaclass=Meta):
    image_dir: ClassVar[Path]
    microscope_position: ClassVar[int]
    microscope_end: ClassVar[int]
    microscope_start: ClassVar[int]
    curr_device: ClassVar[Union[bmscam.BmscamDeviceV2, None]]
    resolution_idx: ClassVar[Union[int, None]]
    imgWidth: ClassVar[Union[int, None]]
    imgHeight: ClassVar[Union[int, None]]
    pData: ClassVar[Union[bytes, None]]
    camera: ClassVar[Union[bmscam.Bmscam, None]]
    recording: ClassVar[bool]
    start_camera_and_motor: ClassVar[bool]
    real_motor_position: ClassVar[int]
    isGPIO: ClassVar[bool]
    motor: ClassVar[abs_motor_type]
    server: ClassVar[serving.BaseWSGIServer]
    image_count: ClassVar[int]
