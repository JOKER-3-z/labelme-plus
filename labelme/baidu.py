from aip import AipBodyAnalysis
import time

""" 你的 APPID AK SK """
APP_ID = '23576996'
API_KEY = '3mKXQH8iqpB0e0F6My0sfWUm'
SECRET_KEY = 'eWLWgmdOFKG8N6vGNTXdTjAQgknyU5ql'

client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)

visible_threshold = 0.23


def get_annoatation_from_baidu(image):
    while True:
        try:
            """ 调用人体关键点识别 """
            res = client.bodyAnalysis(image)
            break
        except Exception as e:
            print("get error", e)
            time.sleep(5)
    return res

