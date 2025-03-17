import settings
from Models.OCR import easyocr
from Models.OCR import got_ocr20
from Models.Multi import phi4
from windowcapture import WindowCapture

ocr = None

def init_ocr_model():
    global ocr
    if ocr is None:
        ocr_type = settings.GetOption("ocr_type")
        ocr_ai_device = settings.GetOption("ocr_ai_device")
        ocr_precision = settings.GetOption("ocr_precision")
        match ocr_type:
            case "easyocr":
                ocr = easyocr.EasyOcr()
            case "got_ocr_20":
                ocr = got_ocr20.Got_ocr_20(ocr_ai_device)
            case "phi4":
                ocr = phi4.Phi4(
                    device=ocr_ai_device,
                    compute_type=ocr_precision
                )

def get_installed_language_names():
    global ocr
    if ocr is not None:
        try:
            if hasattr(ocr, 'get_installed_language_names'):
                return ocr.get_installed_language_names()
            if hasattr(ocr, 'get_languages'):
                return ocr.get_languages()
        except Exception as e:
            print(str(e).encode('utf-8', 'ignore').decode('utf-8', 'ignore'))
    return []

def run_image_processing_from_image(image_src, src_languages):
    global ocr
    if ocr is not None:
        try:
            return ocr.run_image_processing_from_image(image_src, src_languages)
        except Exception as e:
            print(str(e).encode('utf-8', 'ignore').decode('utf-8', 'ignore'))
    return [], None, []

def initialize_window_capture(window_name):
    win_cap = WindowCapture(window_name)
    return win_cap

def run_image_processing(window_name, src_languages):
    global ocr
    screenshot_png = None
    result_lines = []
    bounding_boxes = []
    if ocr is not None:
        try:
            win_cap = initialize_window_capture(window_name)

            # get an updated image of the game
            screenshot, screenshot_png = win_cap.get_screenshot_mss()
            if screenshot is None:
                return None, None, None

            # unitialize
            win_cap.unitialize()

            result_lines, _, bounding_boxes = ocr.run_image_processing_from_image(screenshot, src_languages)
        except Exception as e:
            print(str(e).encode('utf-8', 'ignore').decode('utf-8', 'ignore'))
    return result_lines, screenshot_png, bounding_boxes
