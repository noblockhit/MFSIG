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
            possible_hints = hint.__dict__.get("__args__")
        else:
            possible_hints = [hint]

        print(class_hint, possible_hints)

        val_type = type(__value)
        
        for _hint in possible_hints:
            if isinstance(_hint, types.FunctionType) or isinstance(_hint, types.LambdaType) or hasattr(_hint, "__self__"):
                print(_hint, val_type, __value)
            else:
                if (isinstance(__value, _hint) or (__value is None and can_be_none)) or (ABSType in _hint.__bases__ and val_type.__name__ == _hint.__name__):
                        return super().__setattr__(__name, __value)
        raise ValueError(f"The property {__name} only takes {possible_hints}, got {val_type} <{__value}> instead.")

abs_motor_type = type("Motor", (ABSType,), dict())
abs_camera_type = type("Camera", (ABSType,), dict())

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
    camera: ClassVar[Union[bmscam.Bmscam, abs_camera_type, None]]
    recording: ClassVar[bool]
    start_motor_and_prepare_recording: ClassVar[bool]
    real_motor_position: ClassVar[int]
    isGPIO: ClassVar[bool]
    motor: ClassVar[abs_motor_type]
    server: ClassVar[serving.BaseWSGIServer]
    image_count: ClassVar[int]
    final_image_dir: ClassVar[Path]
    with_bms_cam: ClassVar[bool]
    bms_enum: ClassVar[list]
