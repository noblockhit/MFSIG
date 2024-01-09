import cv2


class Slider:
    sliders = []
    @staticmethod
    def mousewheel(event, x, y, flags, param):
        # print(event, x, y, flags, param)
        for sl in Slider.sliders:
            if sl.x <= x <= sl.x + sl.width and sl.y <= y <= sl.y + sl.height:
                if event == cv2.EVENT_MOUSEWHEEL:
                    if flags > 0:
                        sl.value = min(sl.max_val, sl.value + sl.step_size)
                    else:
                        sl.value = max(sl.min_val, sl.value - sl.step_size)
                
                elif event == cv2.EVENT_LBUTTONDOWN:
                    percentage = (x - sl.x) / sl.width
                    exact_amout = percentage * sl.max_val
                    
                    sl.value = min(sl.max_val, max(sl.min_val, round(exact_amout / sl.step_size) * sl.step_size))
                    
                break
                    
        
    def __init__(self, min_val, max_val, step_size, x, y, width, height, name):
        self.min_val = min_val
        self.max_val = max_val
        self.step_size = step_size
        self.value = self.min_val
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.name = name
        Slider.sliders.append(self)
    
    
    def draw(self, surface):
        n_text_width, n_text_height = cv2.getTextSize(self.name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        n_text_y = int(self.y + n_text_height*1.25)
        n_text_x = int(self.x - n_text_width - 5)
        cv2.putText(surface, self.name, (n_text_x, n_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.rectangle(surface, (self.x, self.y), (self.x + self.width, self.y + self.height), (0, 0, 0), -1)
        cv2.rectangle(surface, (self.x, self.y), (self.x + int(self.value/self.max_val* self.width), self.y + self.height), (255, 0, 0), -1)
        v_text_width, v_text_height = cv2.getTextSize(str(self.value), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        v_text_y = int(self.y + v_text_height*1.25)
        v_text_x = int(self.x + self.value/self.max_val* self.width)
        
        if self.value/self.max_val >= .5:
            v_text_x -= v_text_width

        cv2.putText(surface, str(self.value), (v_text_x, v_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def main():
    sl = Slider(25, 100, 5, 250, 5, 300, 20, "Test Slider")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        sl.draw(frame)
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', sl.mousewheel)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()