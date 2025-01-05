import csv
import threading
from datetime import datetime


def safe_decode(data):
    encodings = ['utf-8', 'utf-16', 'gbk', 'iso-8859-1', 'iso-8859-5', 'iso-8859-6', 'big5', 'shift_jis', 'euc-kr', 'euc-jp', 'windows-1252', 'windows-1251', 'windows-1256']
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode('utf-8', 'replace')  # Default to utf-8 with replacement


# Convert bytes to string recursively over a dictionary or list and decode bytes safely into a string
def handle_bytes(obj):
    if isinstance(obj, bytes):
        return safe_decode(obj)
    elif isinstance(obj, list):
        return [handle_bytes(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: handle_bytes(value) for key, value in obj.items()}
    else:
        return obj


def ns_to_datetime(ns, formatting='%Y-%m-%d %H:%M:%S.%f'):
    # Convert nanoseconds to seconds
    seconds = ns / 1_000_000_000
    # Create a datetime object
    dt_object = datetime.fromtimestamp(seconds)
    # Format the datetime object as a string
    return dt_object.strftime(formatting)[:-3]  # trimming microseconds to milliseconds


transcriptions_list = {}
# Lock for thread-safe dictionary update
transcriptions_list_lock = threading.Lock()

def add_transcription(start_time, end_time, transcription, translation, continous_text=False, file_path=None):
    global transcriptions_list

    start_time_str = ns_to_datetime(start_time)
    end_time_str = ns_to_datetime(end_time)

    # Update the dictionary
    with transcriptions_list_lock:
        transcriptions_list[(start_time, end_time)] = {"transcription": transcription, "translation": translation}

        # Add the new entry to the CSV file
        if file_path is not None and isinstance(file_path, str) and file_path != "":
            with open(file_path, "a", newline='') as transcription_file:
                if continous_text:
                    text_to_append = translation if translation else transcription
                    transcription_file.write(f" {text_to_append}")
                else:
                    csv_writer = csv.writer(transcription_file, quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow([start_time_str, end_time_str, transcription, translation])


def save_transcriptions(file_path: str):
    global transcriptions_list

    with open(file_path, "w", newline='') as transcription_file:
        csv_writer = csv.writer(transcription_file, quoting=csv.QUOTE_MINIMAL)

        # Write headers if you want (optional)
        # csv_writer.writerow(["Start Time", "End Time", "Transcription", "Translation"])

        for (start_time, end_time), entry in transcriptions_list.items():
            transcription = entry["transcription"]
            translation = entry["translation"]
            start_time_str = ns_to_datetime(start_time)
            end_time_str = ns_to_datetime(end_time)
            csv_writer.writerow([start_time_str, end_time_str, transcription, translation])
        transcription_file.close()
