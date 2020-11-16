# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 20:59:53 2017

@author: Administrator
"""
from collections import deque
import requests
import json
import time
import random
# import pymysql.cursors
import traceback
import urllib.request
import json
import requests
import os
import numpy as np
from weibo_test import get_userInfo,make_model,get_weibo, check_interest
from album_downloader import Get_imge,Get_page
import torch
# import multiprocessing
# from  multiprocessing import Pool
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

proxy_addr = "122.241.72.191:808"
def crawlDetailPage(data,queue,recorded):
    content = data['data']['cards']
    #print(content)
    id_list=[]
    for i in content:
        followingId = str(i['user']['id'])
        followingName = i['user']['screen_name']
        followingUrl = i['user']['profile_url']
        followersCount = i['user']['followers_count']
        followCount = i['user']['follow_count']
        followingGender=i['user']['gender']
        if 50<followersCount<550 and followCount<500 and followingGender=='f':
            print("UserID:{}".format(followingId))
            if followingId not in recorded:
                queue.append(followingId)
                recorded[followingId]=1
            # print("NickName:{}".format(followingName))
            # print("Url:{}".format(followingUrl))
            # print("followersCount:{}".format(followersCount))
            # print("followCount:{}".format(followCount))
def use_proxy(url, proxy_addr):
    req = urllib.request.Request(url)
    req.add_header("User-Agent",
                   "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.221 Safari/537.36 SE 2.X MetaSr 1.0")
    proxy = urllib.request.ProxyHandler({'http': proxy_addr})
    opener = urllib.request.build_opener(proxy, urllib.request.HTTPHandler)
    urllib.request.install_opener(opener)
    data = urllib.request.urlopen(req).read().decode('utf-8', 'ignore')
    return data
def get_containerid(id):
    url = 'https://m.weibo.cn/api/container/getIndex?type=uid&value=' + id
    proxy_addr = "122.241.72.191:808"
    data = use_proxy(url, proxy_addr)
    content = json.loads(data).get('data')
    containerid=content.get('userInfo').get('profile_url')[-16:]
    return containerid
def bfs(id,queue,recorded):
    flag1=True
    flag2=True
    i=0
    j=0
    while flag1:
        print("crawling FOLLOWERS page {}:".format(i)+"of User "+id)
       # follower page
        url = "https://m.weibo.cn/api/container/getSecond?containerid="+get_containerid(id)+"_-_FOLLOWERS&page=" + str(i)
        req = requests.get(url)
        jsondata = req.text
        data = json.loads(jsondata)
        if data['ok']:
            crawlDetailPage(data,queue,recorded)
            i+=1
        else:
            flag1=False
        # t = random.randint(1,3)
        # # print("sleep:{}s".format(t))
        # time.sleep(t)
    while flag2:
        # fans page
        print("crawling FANS page {}:".format(j)+"of User "+id)
        url = "https://m.weibo.cn/api/container/getSecond?containerid="+get_containerid(id)+"_-_FANS&page=" + str(j)
        req = requests.get(url)
        jsondata = req.text
        data = json.loads(jsondata)
        if data['ok']:
            crawlDetailPage(data,queue,recorded)
            j+=1
        else:
            flag2=False
        # t = random.randint(1,3)
        # print("sleep:{}s".format(t))
        # time.sleep(t)
def Get_images(id,model,device,path_folder):


    print("---------------crawling user"+" ID "+id+"---------")

    headers = {
        'X-Requested-With': 'XMLHttpRequest',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36',
        'referer': 'https://m.weibo.cn',
    }

    page = 0
    while True:
        page += 1
        print('--------------------Downloading '+str(page)+'th page images...-----------')
        o = Get_page(page, id)

        if not o['ok']:

            print('Download of User'+id+' completed!')
            break
        picURLs = Get_imge(o)

        for url in picURLs:
            if url[-4:] != '.jpg':

                continue
            qr = requests.get(url, headers=headers)
            path = path_folder + '/' + id + url.split('/')[-1]

            try:
                # f = open(path, 'ab')
                # f.write(qr.content)
                # f.close()
                check_interest(qr, device, model,path)

            except Exception as e:
                print(url)
                traceback.print_exc()



def oneFunc(id,queue,recorded, model, device, path_folder):
    Get_images(id, model, device, path_folder)
    bfs(id, queue, recorded)
    if len(recorded) < 10000:
       bfs(id, queue, recorded)
    else:
        print('already get enough userids')



if __name__ == "__main__":

    manager=multiprocessing.Manager()
    queue=manager.list()
    completedID=manager.list()
    recorded=manager.dict()
    initialID='5445836078'
    load_path='model19201116.zip'
    recorded[initialID]=1
    queue.append(initialID)
    path_folder = "E:/weibo_project/pred/"
    device = torch.device("cpu")
    model = make_model(load_path, device)

    # for id in queue:
    #     p = multiprocessing.Pool(8)
    #     # p.apply_async(oneFunc,(id,queue,recorded, model, device, path_folder))
    #     if len(recorded) < 10:
    #         p.apply_async(bfs, (id, queue, recorded))
    #
    #     # else:
    #     # #     np.savetxt('recorded.txt', recorded.keys(), fmt='%s')
    #     #     print('already get enough userids')
    #     p.apply_async(Get_images, (id, model, device, path_folder))
    #     p.close()
    #     p.join()
    # for id in queue:
    #     p = multiprocessing.Pool(8)
    #     # p.apply_async(oneFunc,(id,queue,recorded, model, device, path_folder))
    #     if len(recorded) < 1000:
    #         p.apply_async(bfs, (id, queue, recorded))
    #     else:
    #
    #         np.savetxt('recorded'+initialID+str(len(recorded))+'.txt', recorded.keys(), fmt='%s')
    #         print('already get enough userids')
    #         p.terminate()
    #         break
    #
    #     p.close()
    #     p.join()

    id_list=np.loadtxt('recorded.txt',dtype='str').tolist()
    index=id_list.index('2785582411')
    id_list=id_list[index+1:]
    for id in id_list:
        p = multiprocessing.Pool(8)
        p.apply_async(Get_images, (id, model, device, path_folder))
        p.close()
        p.join()

