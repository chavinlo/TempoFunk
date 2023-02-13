# instrucciones:
# 1. ir al perfil de TikTok deseado
# 2. abrir el devtools
# 3. en Network, filtrar por "?aid=1988"
# 4. solo aplican las requests despues de la finalizacion de carga base de la pagina.
# 5. guardar el JSON recivido desde ahi entonces. (ejemplo, copiar como cURL y guardarlo)
# 6. guardar los JSONs en JSON_FOLDER

import json
import os

JSON_FOLDER = "/workspace/TempoFunk/data/tiktok/collector/jsons"
RAW_OUT_FOLDER = "/workspace/TempoFunk/data/raw"
LIST_TXT_FILE = "/workspace/TempoFunk/data/tiktok/list.txt"

for json_file in os.listdir(JSON_FOLDER):
    print(json_file)
    data = json.load(open(os.path.join(JSON_FOLDER, json_file), "r"))
    for item in data['itemList']:
        post_id = item['id']

        post_folder = os.path.join(RAW_OUT_FOLDER, post_id)
        os.makedirs(post_folder, exist_ok=True)
        post_desc = item['desc']
        post_auth = {"name": item['author']['nickname'], "user_id": item['author']['id'], "unique_id": item['author']['uniqueId']}
        song_data = {"title": item['music']['title'], "playUrl": item['music']['playUrl'], "audio_id": item['music']['id']}
        post_json = {
            "post_id": post_id,
            "post_desc": post_desc,
            "post_auth": post_auth,
            "song_data": song_data
        }
        json.dump(post_json, open(os.path.join(post_folder, "meta_scrap.json"), "w"))
        predict_url = f'https://www.tiktok.com/@{post_auth["unique_id"]}/video/{post_id}'
        open(LIST_TXT_FILE, 'a').write(f'{predict_url}\n')
