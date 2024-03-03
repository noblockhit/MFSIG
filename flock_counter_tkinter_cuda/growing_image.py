import customtkinter as CTk
from tkinter.constants import *
from PIL import ImageTk, Image
import time


CTk.set_appearance_mode("System")
CTk.set_default_color_theme("dark-blue")

class GrowingImage(CTk.CTkCanvas):
    def __init__(self, parent, *args, zoom_factor=False, image=None, **kw):
        CTk.CTkCanvas.__init__(self, parent, *args, bg="gray13", bd=0, highlightthickness=0, relief='ridge', **kw)
        self.parent = parent
        self.zoom_factor = zoom_factor
        self.bind('<Configure>', self._config_size)
        self.src_img = image
        self.pil_img = Image.fromarray(image)
        self.box_width = -1
        self.box_height = -1
        self.box_aspect_ratio = self.box_height / self.box_width
        self.new_image_width = -1
        self.new_image_height = -1
        self.mouse_src_img_x = -1
        self.mouse_src_img_y = -1
        self.mouse_x = -1
        self.mouse_y = -1

        self.src_aspect_ratio = self.src_img.shape[0] / self.src_img.shape[1]
        self.bind("<Enter>", self._set_mouseover_true)
        self.bind("<Leave>", self._set_mouseover_false)

        self.bind("<Motion>", self._mouse_motion)
        self.is_mouse_over = False

        if self.zoom_factor is not None:
            self.bind("<MouseWheel>", self._on_mousewheel)
            self.zoom_amount = 1
            self.zoom_x_offset = 0
            self.zoom_y_offset = 0

    @property
    def img(self):
        return self.src_img
    
    @img.setter
    def img(self, image):
        self.src_img = image
        self.src_aspect_ratio = self.src_img.shape[0] / self.src_img.shape[1]
        self.pil_img = Image.fromarray(self.src_img)

        self._config_size()


    def _mouse_motion(self, event):
        if not self.is_mouse_over:
            return
        
        box_x, box_y = event.x, event.y
        
        x_offset = (self.box_width - self.new_image_width) // 2
        y_offset = (self.box_height - self.new_image_height) // 2

        ## pointer outside of image
        if box_x < x_offset or self.box_width - x_offset < box_x:
            return
        
        if box_y < y_offset or self.box_height - y_offset < box_y:
            return
        
        self.mouse_x = box_x - x_offset
        self.mouse_y = box_y - y_offset
        
        self.mouse_src_img_x = int((self.mouse_x / self.new_image_width) * self.src_img.shape[0])
        self.mouse_src_img_y = int((self.mouse_y / self.new_image_height) * self.src_img.shape[1])


    def _set_mouseover_true(self, event):
        self.is_mouse_over = True


    def _set_mouseover_false(self, event):
        self.is_mouse_over = False


    def _on_mousewheel(self, event):
        img_mouse_x_portion = self.mouse_x / self.new_image_width
        img_mouse_y_portion = self.mouse_y / self.new_image_height

        if self.is_mouse_over:
            prev_attrs = self.zoom_amount, self.zoom_x_offset, self.zoom_y_offset
            if event.state == 4:
                if event.delta > 0:
                    if self.zoom_amount * self.src_img.shape[0] < 5 > self.zoom_amount * self.src_img.shape[1]:
                        return
                    self.zoom_x_offset += self.src_img.shape[1] * self.zoom_amount * img_mouse_x_portion * (1-self.zoom_factor)
                    self.zoom_y_offset += self.src_img.shape[0] * self.zoom_amount * img_mouse_y_portion * (1-self.zoom_factor)
                    self.zoom_amount = self.zoom_amount * self.zoom_factor
                else:
                    self.zoom_amount = min(1, self.zoom_amount * (1/self.zoom_factor))
                    self.zoom_x_offset -= self.src_img.shape[1] * self.zoom_amount * (1-self.zoom_factor) * img_mouse_x_portion
                    self.zoom_y_offset -= self.src_img.shape[0] * self.zoom_amount * (1-self.zoom_factor) * img_mouse_y_portion
                
                
            elif event.state == 0:
                if event.delta > 0:
                    self.zoom_y_offset -= 50 * self.zoom_amount**.5
                else:
                    self.zoom_y_offset += 50 * self.zoom_amount**.5

            elif event.state == 1:
                if event.delta > 0:
                    self.zoom_x_offset -= 50 * self.zoom_amount**.5
                else:
                    self.zoom_x_offset += 50 * self.zoom_amount**.5
                    
            self.zoom_x_offset = max(0, self.zoom_x_offset)
            self.zoom_y_offset = max(0, self.zoom_y_offset)
            self.zoom_x_offset = min(-self.src_img.shape[1] * self.zoom_amount + self.src_img.shape[1], self.zoom_x_offset)
            self.zoom_y_offset = min(-self.src_img.shape[0] * self.zoom_amount + self.src_img.shape[0], self.zoom_y_offset)
            if prev_attrs != (self.zoom_amount, self.zoom_x_offset, self.zoom_y_offset):
                self._redraw_image()


    def _config_size(self, event=None):
        if event is not None:
            self.box_width = event.width
            self.box_height = event.height
        self.box_aspect_ratio = self.box_height / self.box_width

        if self.box_aspect_ratio > self.src_aspect_ratio:
            new_width = self.box_width
            new_height = new_width * self.src_aspect_ratio
        else:
            new_height = self.box_height
            new_width = new_height / self.src_aspect_ratio

        self.new_image_width = int(new_width)
        self.new_image_height = int(new_height)

        self._redraw_image()

    def _redraw_image(self):
        self.img_cropped = self.pil_img.crop((self.zoom_x_offset,
                                              self.zoom_y_offset,
                                              self.src_img.shape[1] * self.zoom_amount + self.zoom_x_offset,
                                              self.src_img.shape[0] * self.zoom_amount + self.zoom_y_offset))
        
        self.image = ImageTk.PhotoImage(self.img_cropped.resize((self.new_image_width, self.new_image_height)))
        
        self.create_image(
            self.box_width // 2,
            self.box_height // 2,
            anchor="center",
            image = self.image
        )