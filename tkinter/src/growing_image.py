import customtkinter as CTk
from tkinter.constants import *
from PIL import ImageTk, Image
import time


CTk.set_appearance_mode("System")
CTk.set_default_color_theme("dark-blue")

class GrowingImage(CTk.CTkCanvas):
    def __init__(self, parent, *args, zoom=False, image=None, **kw):
        CTk.CTkCanvas.__init__(self, parent, *args, bg="gray13", bd=0, highlightthickness=0, relief='ridge', **kw)
        self.parent = parent
        self.bind('<Configure>', self._config_size)
        self.src_img = image
        self.pil_img = Image.fromarray(image)
        self.box_width = -1
        self.box_height = -1
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

        if zoom:
            self.bind("<MouseWheel>", self._on_mousewheel)
            self.zoom_amount = 1
            self.zoom_x_offset = 0
            self.zoom_y_offset = 0



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
        ZOOM_FACTOR = .5

        img_mouse_x_portion = self.mouse_x / self.new_image_width
        img_mouse_y_portion = self.mouse_y / self.new_image_height

        if self.is_mouse_over and event.state == 4:
            prev_zoom_amount = self.zoom_amount
            if event.delta > 0:
                if self.zoom_amount * self.src_img.shape[0] < 5 > self.zoom_amount * self.src_img.shape[1]:
                    return
                self.zoom_x_offset += self.src_img.shape[1] * self.zoom_amount * ZOOM_FACTOR * img_mouse_x_portion
                self.zoom_y_offset += self.src_img.shape[0] * self.zoom_amount * ZOOM_FACTOR * img_mouse_y_portion
                self.zoom_amount = self.zoom_amount * ZOOM_FACTOR
            else:
                self.zoom_amount = min(1, self.zoom_amount * (1/ZOOM_FACTOR))
                self.zoom_x_offset -= self.src_img.shape[1] * self.zoom_amount * ZOOM_FACTOR * img_mouse_x_portion
                self.zoom_y_offset -= self.src_img.shape[0] * self.zoom_amount * ZOOM_FACTOR * img_mouse_y_portion
            
            if self.zoom_amount >= 1:
                self.zoom_x_offset = 0
                self.zoom_y_offset = 0
            
            if prev_zoom_amount!= self.zoom_amount:
                self._redraw_image()
        


    def _config_size(self, event):
        self.box_width = event.width
        self.box_height = event.height
        box_aspect_ratio = event.height / event.width

        if box_aspect_ratio > self.src_aspect_ratio:
            new_width = event.width
            new_height = new_width * self.src_aspect_ratio
        else:
            new_height = event.height
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