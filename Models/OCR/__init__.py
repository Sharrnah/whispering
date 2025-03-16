import settings
from Models.OCR import easyocr
from Models.OCR import got_ocr20

ocr = None

def init_ocr_model():
    global ocr
    if ocr is None:
        ocr_type = settings.GetOption("ocr_type")
        match ocr_type:
            case "easyocr":
                ocr = easyocr.EasyOcr()
            case "got_ocr_20":
                ocr = got_ocr20.Got_ocr_20()

def get_installed_language_names():
    global ocr
    if ocr is not None:
        try:
            return ocr.get_installed_language_names()
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

def run_image_processing(window_name, src_languages):
    global ocr
    if ocr is not None:
        try:
            return ocr.run_image_processing(window_name, src_languages)
        except Exception as e:
            print(str(e).encode('utf-8', 'ignore').decode('utf-8', 'ignore'))
    return [], None, []
