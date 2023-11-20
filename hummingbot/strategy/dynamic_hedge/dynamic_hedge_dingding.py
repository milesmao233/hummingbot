import base64
import hashlib
import hmac
import json
import time
from datetime import datetime
from urllib import parse

import requests


def cal_timestamp_sign(secret):
    # 根据钉钉开发文档，修改推送消息的安全设置https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq
    # 也就是根据这个方法，不只是要有robot_id，还要有secret
    # 当前时间戳，单位是毫秒，与请求调用时间误差不能超过1小时
    # python3用int取整
    timestamp = int(round(time.time() * 1000))
    # 密钥，机器人安全设置页面，加签一栏下面显示的SEC开头的字符串
    secret_enc = bytes(secret.encode('utf-8'))
    string_to_sign = '{}\n{}'.format(timestamp, secret)
    string_to_sign_enc = bytes(string_to_sign.encode('utf-8'))
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    # 得到最终的签名值
    sign = parse.quote_plus(base64.b64encode(hmac_code))
    return str(timestamp), str(sign)


def send_msg(content, robot_id, secret):
    try:
        msg = {
            "msgtype": "text",
            "text": {"content": content + '\n' + datetime.now().strftime("%m-%d %H:%M:%S")}}
        headers = {"Content-Type": "application/json;charset=utf-8"}
        # https://oapi.dingtalk.com/robot/send?access_token=XXXXXX&timestamp=XXX&sign=XXX
        timestamp, sign_str = cal_timestamp_sign(secret)
        url = 'https://oapi.dingtalk.com/robot/send?access_token=' + robot_id + \
              '&timestamp=' + timestamp + '&sign=' + sign_str
        body = json.dumps(msg)
        requests.post(url, data=body, headers=headers, timeout=10)
        print('成功发送钉钉', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        print("发送钉钉失败:", e, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class HedgeDataDingdingRobot:
    def __init__(self, dingding_robot_id, dingding_secret, dingding_waiting_robot_id, dingding_waiting_secret):

        self.dingding_api = {
            'robot_id': dingding_robot_id,
            'secret': dingding_secret,
        }
        self.dingding_api_waiting = {
            'robot_id': dingding_waiting_robot_id,
            'secret': dingding_waiting_secret,
        }

    def send_dingding_msg(self, content):
        """
        :param content:
        :param robot_id:  你的access_token，即webhook地址中那段access_token。
                            例如如下地址：https://oapi.dingtalk.com/robot/send?access_token=81a0e96814b4c8c3132445f529fbffd4bcce66
        :param secret: 你的secret，即安全设置加签当中的那个密钥
        :return:
        """

        robot_id = self.dingding_api['robot_id']
        secret = self.dingding_api['secret']

        send_msg(content, robot_id, secret)

    def send_dingding_waiting_msg(self, content):
        """
        :param content:
        :param robot_id:  你的access_token，即webhook地址中那段access_token。
                            例如如下地址：https://oapi.dingtalk.com/robot/send?access_token=81a0e96814b4c8c3132445f529fbffd4bcce66
        :param secret: 你的secret，即安全设置加签当中的那个密钥
        :return:
        """

        robot_id = self.dingding_api_waiting['robot_id']
        secret = self.dingding_api_waiting['secret']

        send_msg(content, robot_id, secret)
