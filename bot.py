import tweepy
import urllib.request
import numpy as np
import cv2
import os
from os import environ
import time


def setup_api():
    API_KEY = environ['CONSUMER_KEY']
    API_SECRET_KEY = environ['CONSUMER_SECRET']
    ACCESS_TOKEN = environ['ACCESS_KEY']
    ACCESS_TOKEN_SECRET = environ['ACCESS_SECRET']

    auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)
    return api


def process_image(img):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('./assets/data/haarcascade_frontalface_default.xml')
    # Convert image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Loading mask image
    mask = cv2.imread('./assets/img/mask.jpg')

    # Pasting mask image in original image
    for (x, y, w, h) in faces:
        x_mouth = int(x+w/5)
        y_mouth = int(y+h/2)
        w_mouth = int(3*w/5)
        h_mouth = int(3*h/5)

        mask_rescaled = cv2.resize(mask, (w_mouth, h_mouth), interpolation = cv2.INTER_AREA)
        
        img[y_mouth:y_mouth+h_mouth, x_mouth:x_mouth+w_mouth][mask_rescaled != [255, 255, 255]] = mask_rescaled[mask_rescaled != [255, 255, 255]]
        
    return img


def reply_mention(mention):
    if ('media' in mention.entities):
        for media in mention.entities['media']:
            if(media['type'] == 'photo'):
                url = media['media_url']
                
                with urllib.request.urlopen(url) as img_url:
                    arr = np.asarray(bytearray(img_url.read()), dtype=np.uint8)
                    img = cv2.imdecode(arr, -1) # 'Load it as it is'
                    new_img = process_image(img)
                filename = "tmp/{}.png".format(mention.id)
                cv2.imwrite(filename, new_img)
                api.update_with_media(filename, status='', in_reply_to_status_id=mention.id, auto_populate_reply_metadata=True)
                os.remove(filename)
                return

api = setup_api()
while(True):
    with open('last_id', 'r') as file:
        last_id = file.readline()

    if (last_id != ''):
        mentions = api.mentions_timeline(last_id, tweet_mode='extended')
    else :
        mentions = api.mentions_timeline(tweet_mode='extended')

    for mention in mentions:
        reply_mention(mention)

        last_id = mention.id
        with open('last_id', 'w+') as file:
            file.write(str(last_id))
    
    time.sleep(3600)