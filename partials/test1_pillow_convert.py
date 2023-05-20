import sys
import pygame
import numpy as np
from PIL import Image
import time

import bmscam

from sbNative.debugtools import log

bms_enum = bmscam.Bmscam.EnumV2()

def list_devices():
    for idx, device in enumerate(bms_enum):
        log(idx, device.id)

def handleImageEvent(camera, pData):
    try:
        camera.PullImageV3(pData, 0, 24, 0, None)
        print("pulled img")
    except bmscam.HRESULTException as e:
        log(e)
        print("Couldn't pull Still image")


def handleStillImageEvent():
    info = bmscam.BmscamFrameInfoV3()
    try:
        camera.PullImageV3(None, 1, 24, 0, info)
    except bmscam.HRESULTException as e:
        log(e)
        print("Couldn't pull Still image")
    else:
        if info.width > 0 and info.height > 0:
            buff = bytes(bmscam.TDIBWIDTHBYTES(info.width * 24) * info.height)
            try:
                camera.PullImageV3(buff, 1, 24, 0, info)
            except bmscam.HRESULTException:
                pass
            else:
                pil_image = Image.frombytes("RGB", (imgWidth, imgHeight), buff)
                pil_image.save("Image.png")
        

def event_callback(n_event, ctx):
    camera, pData = ctx
    if camera:
        if bmscam.BMSCAM_EVENT_IMAGE == n_event:
            handleImageEvent(camera, pData)

        elif bmscam.BMSCAM_EVENT_EXPOSURE == n_event:
            # self.handleExpoEvent()
            pass
        elif bmscam.BMSCAM_EVENT_TEMPTINT == n_event:
            # self.handleTempTintEvent()
            pass
        elif bmscam.BMSCAM_EVENT_STILLIMAGE == n_event:
            handleStillImageEvent()
            pass
        elif bmscam.BMSCAM_EVENT_ERROR == n_event:
            camera.Close()
            log("Warning", "Generic Error.")
        elif bmscam.BMSCAM_EVENT_DISCONNECTED == n_event:
            camera.Close()
            log("Warning", "Camera disconnect.")

list_devices()
idx = 0
curr_device = bms_enum[idx]

camera = bmscam.Bmscam.Open(curr_device.id)

if camera:
    camera.put_eSize(0)
    resolution = camera.get_eSize()
    imgWidth = curr_device.model.res[resolution].width
    imgHeight = curr_device.model.res[resolution].height

    print(imgWidth, imgHeight)
    preview_resos = [(curr_device.model.res[i].width, curr_device.model.res[i].height) for i in range(0, curr_device.model.preview)]

    camera.put_Option(bmscam.BMSCAM_OPTION_BYTEORDER, 0) #Qimage use RGB byte order
    camera.put_AutoExpoEnable(1)

    pData = bytes(bmscam.TDIBWIDTHBYTES(imgWidth * 24) * imgHeight)

    uimin, uimax, uidef = camera.get_ExpTimeRange()
    usmin, usmax, usdef = camera.get_ExpoAGainRange()
    ## handle exposure evt
    if curr_device.model.flag & bmscam.BMSCAM_FLAG_MONO == 0:
        ## handleTempTintEvent()
        pass
    try:
        camera.StartPullModeWithCallback(event_callback, (camera, pData))
    except bmscam.HRESULTException:
        log("Failed to start camera.")
        camera.Close()
    else:
        bAuto = camera.get_AutoExpoEnable()

    log(sys.getsizeof(pData))


    time.sleep(1)
    pygame.init()


    with open("output_bytestream.txt", "wb") as wf:
        wf.write(pData)

    display = pygame.display.set_mode((imgWidth, imgHeight))
    running = True
    while running:
        start = time.time()
        img_array = np.array(Image.frombytes("RGB", (imgWidth, imgHeight), pData))
                    

        surf = pygame.surfarray.make_surface(img_array)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    camera.Snap(0)

        display.blit(pygame.transform.rotate(surf, 90), (0, 0))
        pygame.display.update()
        print(1/(time.time() - start))
