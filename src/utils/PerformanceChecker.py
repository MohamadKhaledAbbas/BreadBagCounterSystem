import cv2

from src.utils.AppLogging import logger

def run_with_timing(function_name, callback, *args, **kwargs):
    t1 = cv2.getTickCount()
    result = callback(*args, **kwargs)
    t2 = cv2.getTickCount()
    latency = (t2 - t1) * 1000 / cv2.getTickFrequency()
    logger.debug(f"processing time {function_name}: {latency:.2f} ms")
    return result
