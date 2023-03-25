import numpy as np
import win32gui, win32ui, win32con, win32com.client
import mss.tools
import time


class WindowCapture:
    w = 0
    h = 0
    hwnd = None
    window_name = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    def __init__(self, window_name=None):
        self.shell = win32com.client.Dispatch("WScript.Shell")
        self.window_name = window_name
        self.hwnd = None

        # find the handle for the window we want to capture.
        # if no window name is given, capture the entire screen
        if window_name is None:
            self.hwnd = win32gui.GetDesktopWindow()
        else:
            #self.hwnd = win32gui.FindWindow(None, window_name)
            self.hwnd = self.find_window_by_title(window_name)

            if not self.hwnd:
                self.hwnd = win32gui.GetDesktopWindow()
                print('Window not found: {}. capturing whole desktop instead.'.format(window_name))
                # raise Exception('Window not found: {}'.format(window_name))
            else:
                print('Window found: {}'.format(window_name))
                # try to focus window to be able to capture it
                try:
                    self.bring_to_top()
                    self.set_act_win()
                    self.set_as_foreground_window()
                    time.sleep(0.2)
                except Exception as e:
                    print('Failed to focus window: {}'.format(window_name))

        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        # account for the window border and titlebar and cut them off
        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

        # set the cropped coordinates offset so we can translate screenshot
        # images into actual screen positions
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

    def find_window_by_title(self, title):
        windows = []

        def enum_callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if window_title.lower() == title.lower():
                    windows.append(hwnd)
            return True

        win32gui.EnumWindows(enum_callback, None)
        return windows[0] if windows else None

    def bring_to_top(self):
        win32gui.BringWindowToTop(self.hwnd)

    def set_as_foreground_window(self):
        self.shell.SendKeys('%')
        win32gui.SetForegroundWindow(self.hwnd)

    def set_act_win(self):
        win32gui.SetActiveWindow(self.hwnd)

    def get_screenshot(self):

        # get the window image data
        w_dc = win32gui.GetWindowDC(self.hwnd)
        dc_obj = win32ui.CreateDCFromHandle(w_dc)
        c_dc = dc_obj.CreateCompatibleDC()
        data_bit_map = win32ui.CreateBitmap()
        data_bit_map.CreateCompatibleBitmap(dc_obj, self.w, self.h)
        c_dc.SelectObject(data_bit_map)
        c_dc.BitBlt((0, 0), (self.w, self.h), dc_obj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        # data_bit_map.SaveBitmapFile(c_dc, 'debug.bmp')
        signed_ints_array = data_bit_map.GetBitmapBits(True)
        img = np.fromstring(signed_ints_array, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # free resources
        dc_obj.DeleteDC()
        c_dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, w_dc)
        win32gui.DeleteObject(data_bit_map.GetHandle())

        # drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type()
        #   && _img.dims() <= 2 in function 'cv::matchTemplate'
        img = img[..., :3]

        # make image C_CONTIGUOUS to avoid errors that look like:
        #   File ... in draw_rectangles
        #   TypeError: an integer is required (got type tuple)
        # see the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
        img = np.ascontiguousarray(img)

        return img

    def get_screenshot_mss(self):
        with mss.mss() as sct:
            # The screen part to capture
            monitor = {"top": self.offset_y, "left": self.offset_x, "width": self.w, "height": self.h}
            # output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)

            # Grab the data
            sct_img = sct.grab(monitor)

            # Convert to numpy array
            img = np.frombuffer(sct_img.raw, dtype='uint8')
            img.shape = (self.h, self.w, 4)

            # Save to the picture file
            png_image = mss.tools.to_png(sct_img.rgb, sct_img.size, output=None)

            # drop alpha channel
            img = img[..., :3]

            # Save to the picture file
            return img, png_image


    # find the name of the window you're interested in.
    # once you have it, update window_capture()
    # https://stackoverflow.com/questions/55547940/how-to-get-a-list-of-the-name-of-every-open-window
    @staticmethod
    def list_window_names():
        window_list = []

        # noinspection PyPep8Naming
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                if win32gui.GetWindowText(hwnd).strip() != '':
                    window_list.append({"hwnd": hex(hwnd), "title": win32gui.GetWindowText(hwnd)})
                # print(hex(hwnd), win32gui.GetWindowText(hwnd))

        win32gui.EnumWindows(winEnumHandler, None)
        return window_list

    # translate a pixel position on a screenshot image to a pixel position on the screen.
    # pos = (x, y)
    # WARNING: if you move the window being captured after execution is started, this will
    # return incorrect coordinates, because the window position is only calculated in
    # the __init__ constructor.
    def get_screen_position(self, pos):
        return pos[0] + self.offset_x, pos[1] + self.offset_y
