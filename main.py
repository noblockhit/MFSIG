from libs import cameraParser
from libs.cameraParser import bmscam


if __name__ == "__main__":
    print("\n-----------------------------------------\n")
    print("The available devices are: \n")
    print(cameraParser.list_devices())
    
    while True:
        curr_device_idx = input("Enter device number: ")

        if not curr_device_idx.isdigit():
            print("That is not a digit!")
            continue
        
        curr_device_idx = int(curr_device_idx)
        if len(cameraParser.bms_enum) <= curr_device_idx:
            print("There is no device corresponding to that number!")
            continue
            
        curr_device = cameraParser.bms_enum[curr_device_idx]
        break
    
    print("\n-----------------------------------------\n")

    print("Your options for resolutions are:", *[f"\nno.: {t[0]}: {t[1][0]}x{t[1][1]}" for t in cameraParser.get_current_devices_resolution_options(curr_device)], "\n")

    while True:
        curr_res_idx = input("Enter resolution number: ")

        if not curr_res_idx.isdigit():
            print("That is not a digit!")
            continue
        
        curr_res_idx = int(curr_res_idx)
        if len(cameraParser.get_current_devices_resolution_options(curr_device)) <= curr_res_idx:
            print("There is no resolution corresponding to that number!")
            continue
        break

    camera = bmscam.Bmscam.Open(curr_device.id)

    if camera:
        camera.put_eSize(curr_res_idx)
        resolution_idx = camera.get_eSize()

        imgWidth = curr_device.model.res[resolution_idx].width
        imgHeight = curr_device.model.res[resolution_idx].height

        print(imgWidth, imgHeight)
        
        camera.put_Option(bmscam.BMSCAM_OPTION_BYTEORDER, 0)
        camera.put_AutoExpoEnable(1)

        pData = bytes(bmscam.TDIBWIDTHBYTES(imgWidth * 24) * imgHeight)
        
        try:
            camera.StartPullModeWithCallback(cameraParser.event_callback, (camera, pData))
        except bmscam.HRESULTException:
            print("Failed to start camera.")
            camera.Close()
        else:
            bAuto = camera.get_AutoExpoEnable()