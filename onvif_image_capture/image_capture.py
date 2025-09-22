#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
from io import BytesIO
from pathlib import Path

from onvif import ONVIFCamera
import requests
from requests.auth import HTTPDigestAuth
from PIL import Image

# ========= Basic Config =========
HOST = "192.168.72.232"
PORT = 80
USER = "admin"
PWD = "1234qwer"

left_top_x = 1919
left_top_y = 550
right_bottom_x = 2909
right_bottom_y = 1419
CROP_BOX = (left_top_x, left_top_y, right_bottom_x, right_bottom_y)

OUTPUT_ORI_DIR = Path("./snapshots_ori")
OUTPUT_CROP_DIR = Path("./snapshots")
OUTPUT_ORI_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CROP_DIR.mkdir(parents=True, exist_ok=True)
# ===========================


def get_snapshot_url(host, port, user, pwd) -> str:
    """
    Obtain the Snapshot URI via ONVIF.
    Failure will result in a fallback to ISAPI.
    """
    cam = ONVIFCamera(host, port, user, pwd)

    dev = cam.create_devicemgmt_service()
    info = dev.GetDeviceInformation()
    print("Device:", info.Manufacturer, info.Model)

    media = cam.create_media_service()
    profiles = media.GetProfiles()
    token = profiles[0].token

    try:
        snap_uri = media.GetSnapshotUri({'ProfileToken': token})
        url = snap_uri.Uri
    except Exception:
        url = f"http://{host}/ISAPI/Streaming/channels/101/picture"

    print("Snapshot URL:", url)
    return url


def fetch_snapshot(url, user, pwd) -> Image.Image:
    """catch JPEG then return PIL.Image"""
    resp = requests.get(url, auth=HTTPDigestAuth(user, pwd), timeout=8)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def save_images(img: Image.Image, crop_box):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # save the ori snapshot
    ori_path = OUTPUT_ORI_DIR / f"snapshot_ori_{ts}.jpg"
    img.save(ori_path, "JPEG", quality=95)
    print(f"Original saved: {ori_path.resolve()}, size = {img.size}")

    # crop the snapshot
    cropped = img.crop(crop_box)
    crop_path = OUTPUT_CROP_DIR / f"snapshot_crop_{ts}.jpg"
    cropped.save(crop_path, "JPEG", quality=95)

    print("Saved original:", ori_path.resolve())
    print("Saved cropped :", crop_path.resolve())


def main():
    url = get_snapshot_url(HOST, PORT, USER, PWD)
    img = fetch_snapshot(url, USER, PWD)
    print("Fetched snapshot size:", img.size)
    save_images(img, CROP_BOX)


if __name__ == "__main__":
    main()
