
import requests
import json


def get_real_rid(rid):
    room_url = 'https://api.live.bilibili.com/room/v1/Room/room_init?id=' + str(rid)
    response = requests.get(url=room_url).json()
    data = response.get('data', 0)
    if data:
        live_status = data.get('live_status', 0)
        room_id = data.get('room_id', 0)
    else:
        live_status = room_id = 0
    return live_status, room_id


def get_real_url(rid):
    # with open("config/config.json", "r", encoding="utf-8") as load_f:
    #     load_dict = json.load(load_f)
    #     cookie = load_dict["bilibili_cookie"]

    headers = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.6,en;q=0.4,zh-TW;q=0.2',
        'Connection': 'keep-alive',
        # 'Cookie': cookie,
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/59.0.3071.115 Safari/537.36 '
    }
    room = get_real_rid(rid)
    live_status = room[0]
    room_id = room[1]
    if live_status:
        try:
            room_url = 'https://api.live.bilibili.com/xlive/web-room/v1/index/getRoomPlayInfo?room_id={}&play_url=1&mask=1&qn=0&platform=web'.format(room_id)
            response = requests.get(url=room_url,headers=headers).json()
            durl = response.get('data').get('play_url').get('durl', 0)
            real_url = durl[-1].get('url')
        except:
            real_url = '疑似部分国外IP无法GET到正确数据，待验证'
    else:
        real_url = '未开播或直播间不存在'
    return real_url
rid = input('请输入bilibili房间号：\n')
real_url = get_real_url(rid)
