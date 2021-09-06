import os
import json
import time
import base64
import requests


class CaptchaCollector:
    def __init__(self, number_of_captcha=360, path='captcha'):
        self.number_of_captcha = number_of_captcha
        self.path = path
        self.create_output_path()
        self.get_captcha()

    def get_captcha_hash(self):
        url = "https://sida.medu.ir/api/Captcha/GetCaptcha"
        response = requests.get(url)
        json_string = response.content
        response_list = json.loads(json_string)
        return response_list['data']['value']

    def check_path_exists(self):
        return os.path.isdir(self.path)

    def create_output_path(self):
        CHECK_FOLDER = self.check_path_exists()
        if not CHECK_FOLDER:
            os.makedirs(self.path)

    def convert_captcha_to_image(self, captcha_hash=None):
        captcha_image_name = time.time_ns()
        captcha_image_name = self.path + "/" + str(captcha_image_name) + ".png"
        if captcha_hash is None:
            captcha_hash = self.get_captcha_hash()
        with open(captcha_image_name, "wb") as capt:
            capt.write(base64.standard_b64decode(captcha_hash))

    def get_captcha(self):
        print("\t+-----------------------------------+")
        for i in range(self.number_of_captcha):
            print('\t|   Fetching {0: >3}`th Captcha ....    |'.format(i + 1))
            self.convert_captcha_to_image()
        print("\t+-----------------------------------+")
        print("\t|   Fetching Complete ...           |")
        print("\t+-----------------------------------+")


cap = CaptchaCollector()
