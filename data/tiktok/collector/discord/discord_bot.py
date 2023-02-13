#Discord bot for data gathering!!!! =)))

import interactions
import json
import requests
import aiohttp
import base64
import asyncio
import io
import os
from utils_db import *
from interactions.ext.files import command_send, command_edit, component_send, component_edit
from concurrent import futures
from interactions.ext.files import command_send, command_edit
from datetime import datetime
import copy

BOT_TOKEN = ""
GUILD_ID = ""
RAW_VIDEO_PATH = "/workspace/TempoFunk/data/raw"

def map_raw_videos(path: str):
    vid_map = []
    list_of_folders = os.listdir(path)
    for folder in list_of_folders:
        predict_video_path = os.path.join(path,folder,"video.webm")
        predict_meta_path = os.path.join(path,folder,"meta_scrap.json")
        video_exists = os.path.exists(predict_video_path)
        metascrap_exists = os.path.exists(predict_meta_path)
        if video_exists is True and metascrap_exists is True:
            vid_map.append({
                "video_path": predict_video_path,
                "meta_path": predict_meta_path,
                "labeled": False
            })
    return vid_map

vid_map = map_raw_videos(RAW_VIDEO_PATH)

bot = interactions.Client(token=BOT_TOKEN)
bot.load('interactions.ext.files')

def make_label_embed(data):
    embed = interactions.Embed()
    embed.add_field("Description", value=data['post_desc'])
    embed.add_field("Username", value=data['post_user'])
    embed.add_field("Song Name", value=data['post_song'])

def sort_list(vid_map: list):
    return sorted(vid_map, key=lambda x: x["labeled"], reverse=True)

def get_task(vid_map: list):
    selected_dict = vid_map[0]
    return selected_dict

@bot.command(
    name="label",
    description="Run this command to get a video to Label.",
    scope=GUILD_ID
)
async def label_request(ctx: interactions.CommandContext):
