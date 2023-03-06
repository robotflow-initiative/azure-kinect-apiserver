# coding=gbk
# https://zhuanlan.zhihu.com/p/359971449,
# https://github.com/dmnfarrell/tkintertable/wiki/Usage
# https://github.com/dmnfarrell/tkintertable/issues/47
import json
import tkinter
# from tkintertable import TableCanvas
from tkinter import *

import requests


def fn_get(string):
    def get():
        r = requests.get(url=f'http://127.0.0.1:5050/{string}', timeout=5)
        if r.status_code == 200:
            result = json.loads(r.content.decode())
            print(f"result = {result}")

    return get


shot_index = 0


def fn_shot():
    def shot():
        global shot_index
        print(f"shot_index = {shot_index}")
        r = requests.get(url=f'http://127.0.0.1:5050/shot/{shot_index}', timeout=15)
        shot_index += 1
        if r.status_code == 200:
            result = json.loads(r.content.decode())
            print(f"result = {result}")

    return shot


if __name__ == '__main__':
    master = tkinter.Tk()
    master.geometry('300x200')

    frame1 = Frame(master)
    frame2 = Frame(master)
    frame3 = Frame(master)
    frame4 = Frame(frame3)
    frame1.pack(padx=1, pady=1, side='top')
    frame2.pack(padx=1, pady=1, side='top')
    frame3.pack(padx=1, pady=1, side='top')
    frame4.pack(padx=1, pady=1, side='right')

    frame4_left = Frame(frame4)
    frame4_right = Frame(frame4)

    frame4_left.pack(padx=1, pady=1, side='left')
    frame4_right.pack(padx=1, pady=1, side='left')

    frame1_left = Frame(frame1)
    frame1_left.pack(side='left')
    frame1_right = Frame(frame1)
    frame1_right.pack(side='left')

    device_list = Button(frame1_right, text="list device", command=fn_get("list"))
    device_list.pack(side='top')
    start = Button(frame1_right, text="start recording",
                   command=fn_get("start"))
    start.pack(side='top')
    stop = Button(frame1_right, text="stop recording", command=fn_get("stop"))
    stop.pack(side='top')
    stop = Button(frame1_right, text="shot", command=fn_shot())
    stop.pack(side='top')
    master.mainloop()
