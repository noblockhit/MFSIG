import customtkinter as CTk
from tkinter.constants import *
import time


CTk.set_appearance_mode("System")
CTk.set_default_color_theme("dark-blue")

class VerticalScrolledFrame(CTk.CTkFrame):
    def __init__(self, parent, *args, **kw):
        CTk.CTkFrame.__init__(self, parent, *args, **kw)

        self.is_mouse_over = False
        self.y_scroll = 0
        self.items = []
        # Create a canvas object and a vertical scrollbar for scrolling it.
        self.yscrollbar = CTk.CTkScrollbar(self)
        self.yscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)
        self.canvas = CTk.CTkCanvas(self, bd=0, highlightthickness=0, 
                                width = 200, height = 300,
                                yscrollcommand=self.yscrollbar.set,
                                bg="gray13")
        
        # parent.bind("<Enter>", self._set_mouseover_true)
        # parent.bind("<Leave>", self._set_mouseover_false)

        self.canvas.bind("<MouseWheel>", self._scroll)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        self.yscrollbar.configure(command = self._scroll)
 
        # Reset the view
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
 
        # Create a frame inside the canvas which will be scrolled with it.
        self.interior = CTk.CTkFrame(self.canvas)
        self.interior.bind('<Configure>', self._configure_interior)
        self.canvas.bind('<Configure>', self._configure_canvas)
        self.interior_id = self.canvas.create_window(0, 0, window=self.interior, anchor=NW)
 
 
    def _configure_interior(self, event):
        # Update the scrollbars to match the size of the inner frame.
        size = (self.interior.winfo_reqwidth(), self.interior.winfo_reqheight())
        self.canvas.config(scrollregion=(0, 0, size[0], size[1]))
        if self.interior.winfo_reqwidth() != self.canvas.winfo_width():
            # Update the canvas's width to fit the inner frame.
            self.canvas.config(width = self.interior.winfo_reqwidth())


    def _configure_canvas(self, event):
        if self.interior.winfo_reqwidth() != self.canvas.winfo_width():
            # Update the inner frame's width to fill the canvas.
            self.canvas.itemconfigure(self.interior_id, width=self.canvas.winfo_width())


    def _scroll(self, *args):
        print("_on_mousewheel", time.time())
        s, e = self.yscrollbar.get()
        
        for a in args:
            print(a)
        if args[0] == "scroll":
            if args[1] > 0:
                if e < 1:
                    self.y_scroll += 1 / len(self.items)
            else:
                if s > 0:
                    self.y_scroll -= 1 / len(self.items)
        elif args[0] == "moveto":
            self.y_scroll = args[1]
        else:
            if args[0].delta < 0:
                if e < 1:
                    self.y_scroll += 1 / len(self.items)
            else:
                if s > 0:
                    self.y_scroll -= 1 / len(self.items)
            
        print(self.y_scroll)
        self.canvas.yview_moveto(self.y_scroll)


    def _set_mouseover_true(self, event):
        print("enter", time.time())
        self.is_mouse_over = True


    def _set_mouseover_false(self, event):
        print("exit", time.time())
        self.is_mouse_over = False
        
    def register_item(self, item):
        item.bind("<MouseWheel>", self._scroll)
        self.items.append(item)

    def unregister_item(self, item):
        self.items.remove(item)
        
    def unregister_all_items(self):
        self.items = []
