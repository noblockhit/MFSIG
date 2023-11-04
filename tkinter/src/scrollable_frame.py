import customtkinter as CTk
from tkinter.constants import *
 
CTk.set_appearance_mode("System")
CTk.set_default_color_theme("dark-blue")

class VerticalScrolledFrame(CTk.CTkFrame):
    def __init__(self, parent, *args, **kw):
        CTk.CTkFrame.__init__(self, parent, *args, **kw)

        self.is_mouse_over = False
 
        # Create a canvas object and a vertical scrollbar for scrolling it.
        vscrollbar = CTk.CTkScrollbar(self)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)
        self.canvas = CTk.CTkCanvas(self, bd=0, highlightthickness=0, 
                                width = 200, height = 300,
                                yscrollcommand=vscrollbar.set,
                                bg="gray13")
        
        self.canvas.bind("<Enter>", self._set_mouseover_true)
        self.canvas.bind("<Leave>", self._set_mouseover_false)

        parent.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        vscrollbar.configure(command = self.canvas.yview)
 
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


    def _on_mousewheel(self, event):
        if self.is_mouse_over and event.state != 4:
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")


    def _set_mouseover_true(self, event):
        self.is_mouse_over = True


    def _set_mouseover_false(self, event):
        self.is_mouse_over = False


if __name__ == '__main__':
    class Window():
        def __init__(self, master, *args, **kwargs):
            self.frame = VerticalScrolledFrame(master)
            self.frame.pack(expand = True, fill = "both")
    
            for i in range(3):
                CTk.CTkButton(self.frame.interior, text=f"Button {i}").pack(padx=10, pady=5)
    
    root = CTk.CTk()
    window = Window(root)
    root.mainloop()
