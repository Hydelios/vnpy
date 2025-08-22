"""
Global setting of the trading platform.
"""

from logging import CRITICAL
from tzlocal import get_localzone_name

from .utility import load_json


SETTINGS: dict = {
    "font.family": "微软雅黑",
    "font.size": 12,

    "log.active": True,
    "log.level": CRITICAL,
    "log.console": True,
    "log.file": True,

    "email.server": "smtp.qq.com",
    "email.port": 465,
    "email.username": "",
    "email.password": "",
    "email.sender": "",
    "email.receiver": "",

    "datafeed.name": "rqdata",
    "datafeed.username": "license",
    "datafeed.password": "NHiG-lJ7WEhZD484_1f0_dpHhGL2THWmakjofFWwPgxNMXAzi7-1Qap6ZgZMeFATqiTJCoXX87oMpHjQHf4OmyUF2xurOM6QA5FRea4h7aSFHjz4dd7iy7Cdmg2qIFarpLKVKwxfYEVpLEWEuaatOaeAsigMDanpC0EkSgzR0Ig=avQbi6gVc9IUXZuYi4aSDQDsaLp5-HzwMPnqTbmkB1bwwSAvYyG6bq5-KS2RbxFgi6vAHZxUv-eUMXQaS3wNxwDAChRdZTjLTLq__Kbr2fytXes2T3DoMTxBhoal-S3WaGfeYJGWgUYqS2SYexqOj_PgPepTURvSbbL4KynjV1U=",

    "database.timezone": get_localzone_name(),
    "database.name": "sqlite",
    "database.database": "database.db",
    "database.host": "",
    "database.port": 0,
    "database.user": "",
    "database.password": ""
}


# Load global setting from json file.
SETTING_FILENAME: str = "vt_setting.json"
SETTINGS.update(load_json(SETTING_FILENAME))
