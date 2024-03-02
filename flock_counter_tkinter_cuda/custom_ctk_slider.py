from customtkinter import CTkSlider
from time import perf_counter_ns


class CCTkSlider(CTkSlider):
    def __init__(self, *args, command=None, **kwargs):
        super(CCTkSlider, self).__init__(*args, command=command, **kwargs)
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.last_scroll_time = -1
        self._cctkcommand = command
            
    def _on_mousewheel(self, event):
        curr_time = perf_counter_ns()*10**-9
        
        m = 1
        if curr_time - self.last_scroll_time < .1:
            m = .1/(curr_time - self.last_scroll_time)
        
        self.set(int(self.get()+(event.delta/120)*m))
        self._cctkcommand(self.get())
        self.last_scroll_time = curr_time