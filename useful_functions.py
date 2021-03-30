from Constantsv2 import *

def convert_to_pixels(length):
    return round(length*PIXEL_CONVERSION_FACTOR)

def clamp(minvalue, value, maxvalue):
    return max(minvalue, min(value, maxvalue))