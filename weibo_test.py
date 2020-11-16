import os
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
import imageio
from PIL import Image
from skimage.transform import resize as imresize
from tqdm import tqdm
img_to_tensor = transforms.ToTensor()
import pickle
import torch.optim as optim
import time
# -*- coding: utf-8 -*-
import urllib.request
import json
import requests
import os
from io import BytesIO
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

path = 'E:/weibo_project/pred/'

proxy_addr = "122.241.72.191:808"
pic_num = 0
weibo_name = "programmer"


def use_proxy(url, proxy_addr):
    req = urllib.request.Request(url)
    req.add_header("User-Agent",
                   "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.221 Safari/537.36 SE 2.X MetaSr 1.0")
    proxy = urllib.request.ProxyHandler({'http': proxy_addr})
    opener = urllib.request.build_opener(proxy, urllib.request.HTTPHandler)
    urllib.request.install_opener(opener)
    data = urllib.request.urlopen(req).read().decode('utf-8', 'ignore')
    return data


def get_containerid(url):
    data = use_proxy(url, proxy_addr)
    content = json.loads(data).get('data')
    for data in content.get('tabsInfo').get('tabs'):
        if (data.get('tab_type') == 'weibo'):
            containerid = data.get('containerid')
    return containerid


def get_userInfo(id):
    url = 'https://m.weibo.cn/api/container/getIndex?type=uid&value=' + id
    data = use_proxy(url, proxy_addr)
    content = json.loads(data).get('data')
    profile_image_url = content.get('userInfo').get('profile_image_url')
    description = content.get('userInfo').get('description')
    profile_url = content.get('userInfo').get('profile_url')
    verified = content.get('userInfo').get('verified')
    guanzhu = content.get('userInfo').get('follow_count')
    name = content.get('userInfo').get('screen_name')
    global weibo_name
    weibo_name=name
    fensi = content.get('userInfo').get('followers_count')
    gender = content.get('userInfo').get('gender')
    urank = content.get('userInfo').get('urank')
    print("NickName：" + name + "\n" + "ProfileURL：" + profile_url + "\n" + "IconURL：" + profile_image_url + "\n" + "Verified：" + str(
        verified) + "\n" + "Intro：" + description + "\n" + "Followering：" + str(guanzhu) + "\n" + "Fans：" + str(
        fensi) + "\n" + "Gender：" + gender + "\n" + "Level：" + str(urank) + "\n")


def get_weibo(id, device,model):

    global pic_num
    i = 1
    while True and i<31:
        url = 'https://m.weibo.cn/api/container/getIndex?type=uid&value=' + id
        weibo_url = 'https://m.weibo.cn/api/container/getIndex?type=uid&value=' + id + '&containerid=' + get_containerid(
            url) + '&page=' + str(i)
        try:
            data = use_proxy(weibo_url, proxy_addr)
            content = json.loads(data).get('data')
            cards = content.get('cards')
            if (len(cards) > 0):
                for j in range(len(cards)):
                    print("-----crawling Page " + str(i) + " No." + str(j) + "------")
                    card_type = cards[j].get('card_type')
                    if (card_type == 9):
                        mblog = cards[j].get('mblog')
                        # attitudes_count = mblog.get('attitudes_count')
                        # comments_count = mblog.get('comments_count')
                        # created_at = mblog.get('created_at')
                        # reposts_count = mblog.get('reposts_count')
                        # scheme = cards[j].get('scheme')
                        # text = mblog.get('text')
                        if mblog.get('pics') != None:
                            # print(mblog.get('original_pic'))
                            # print(mblog.get('pics'))
                            pic_archive = mblog.get('pics')
                            for _ in range(len(pic_archive)):
                                pic_num += 1
                                # print(pic_archive[_]['large']['url'])
                                imgurl = pic_archive[_]['large']['url']

                                if imgurl[-4:]!='.jpg':
                                    continue
                                img = requests.get(imgurl)
                                path_file=path + '/' + str(id) + str(pic_num) + '_' + '.jpg'
                                check_interest(img, device, model,path_file)
                i += 1
            else:
                break
        except Exception as e:
            print(e)
            pass


def check_interest(img, device,model,path_file):
    transform = transforms.Compose(
        [
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    im = Image.open(BytesIO(img.content))
    im = transform(im)
    im = im.to(device)
    outputs = model(im.float()[None, ...])
    m = nn.Softmax()
    outputs = m(outputs).cpu().data.numpy()
    print(outputs[0])
    if outputs[0, 1] > 0.6:
        print('________________find one_______________________________________')
        f = open(path_file,'ab')
        f.write(img.content)
        f.close()

def make_model(load_path, device):
    model = models.vgg16(pretrained=True)

    for param in model.features.parameters():
        param.require_grad = False
    model.classifier[6] = nn.Linear(4096, 2)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model
#
if __name__ == "__main__":
    id=''
    get_userInfo(id)
    device = torch.device("cuda:0")
    load_path='model.zip'
    model = make_model(load_path,device)
    get_weibo(id, device,model)
