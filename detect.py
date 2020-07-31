import io
import json
import cv2
import numpy as np
import requests
from data_preprocess import process_image_for_ocr

def detect(roi):
    # Ocr
    url_api = "https://api.ocr.space/parse/image"
    _, compressedimage = cv2.imencode(".png", roi, [1, 90])
    file_bytes = io.BytesIO(compressedimage)
    result = requests.post(url_api,
                  files = {"screenshot.png": file_bytes},
                  data = {"apikey": "2d53a54b1488957",
                          "language": "eng", "isTable": "true"})

    result = result.content.decode()
    result = json.loads(result)

    parsed_results = result.get("ParsedResults")[0]
    text_overlay = parsed_results.get("TextOverlay")
    line = text_overlay.get("Lines")[0]
    words = line.get("Words")

    top = int(line.get("MinTop"))
    height = int(line.get("MaxHeight"))

    start = words[0]
    s_l = int(start.get("Left"))
    s_t = int(start.get("Top"))
    end = words[-1]
    e_l = int(end.get("Left"))
    e_t = int(end.get("Top"))
    e_w = int(end.get("Width"))
    e_r = int(e_l + e_w)

    padding = 4
    #cv2.circle(img,(s_l,top ),5,(255,255,12),1)
    #cv2.circle(img,(e_r,height + top ),10,(186,187,164),1)
    #cv2.circle(img,(0,0 ),20,(3,3,2),1)
    #cv2.rectangle(img,(int(s_l - padding) , int(top - padding)),(int(e_r + padding), int(height + top + padding)),(36,255,12),1)
    crop = roi[int(top - padding):int(height + top + padding), int(s_l - padding):int(e_r + padding)]

    return crop