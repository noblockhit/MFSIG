import customtkinter as CTk
from tkinter.constants import *
from PIL import ImageTk, Image
from growing_image import GrowingImage

class PreviewImage(GrowingImage):
    def __init__(self, parent, *args, **kw):
        GrowingImage.__init__(self, parent, *args, zoom=True, **kw)

