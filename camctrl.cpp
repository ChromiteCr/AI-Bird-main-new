#include <iostream>
#include <thread>
#include <chrono>

// Fix for macOS: __int64 is a Windows-only type
#define __int64 long long

#include "EDSDK.h"
#include "EDSDKErrors.h"
#include "EDSDKTypes.h"

// 简单错误检查
void checkError(EdsError err, const std::string& msg) {
    if (err != EDS_ERR_OK) {
        std::cerr << msg << " Error Code: 0x" << std::hex << err << std::endl;
        exit(1);
    }
}

// ===================== 关键：Object Event 回调 =====================
EdsError EDSCALLBACK handleObjectEvent(
    EdsObjectEvent event,
    EdsBaseRef object,
    EdsVoid * context)
{
    if (event == kEdsObjectEvent_DirItemCreated)
    {
        EdsDirectoryItemRef dirItem = (EdsDirectoryItemRef)object;

        EdsDirectoryItemInfo dirItemInfo;
        EdsGetDirectoryItemInfo(dirItem, &dirItemInfo);

        std::cout << "[INFO] Downloading: " << dirItemInfo.szFileName << std::endl;

        EdsStreamRef stream = NULL;

        EdsError err = EdsCreateFileStream(
            dirItemInfo.szFileName,
            kEdsFileCreateDisposition_CreateAlways,
            kEdsAccess_ReadWrite,
            &stream);

        if (err == EDS_ERR_OK)
        {
            err = EdsDownload(dirItem, dirItemInfo.size, stream);
        }

        if (err == EDS_ERR_OK)
        {
            err = EdsDownloadComplete(dirItem);
        }

        if (stream) EdsRelease(stream);
        if (dirItem) EdsRelease(dirItem);
    }

    return EDS_ERR_OK;
}

int main()
{
    EdsError err = EDS_ERR_OK;

    // 初始化 SDK
    err = EdsInitializeSDK();
    checkError(err, "Init SDK");

    // 获取相机列表
    EdsCameraListRef cameraList = NULL;
    err = EdsGetCameraList(&cameraList);
    checkError(err, "Get camera list");

    EdsUInt32 count = 0;
    err = EdsGetChildCount(cameraList, &count);
    checkError(err, "Get camera count");

    if (count == 0) {
        std::cerr << "No camera found." << std::endl;
        return 1;
    }

    // 获取第一台相机
    EdsCameraRef camera = NULL;
    err = EdsGetChildAtIndex(cameraList, 0, (EdsBaseRef*)&camera);
    checkError(err, "Get camera");

    // 打开会话
    err = EdsOpenSession(camera);
    checkError(err, "Open session");

    // ===================== 注册事件 =====================
    err = EdsSetObjectEventHandler(
        camera,
        kEdsObjectEvent_All,
        handleObjectEvent,
        NULL);
    checkError(err, "Set Object Event Handler");
    // ===================== 保存到电脑 =====================
    EdsUInt32 saveTo = kEdsSaveTo_Both;
    err = EdsSetPropertyData(
        camera,
        kEdsPropID_SaveTo,
        0,
        sizeof(saveTo),
        &saveTo);
    checkError(err, "Set SaveToHost");

    // ===================== 设置录制槽为 CFexpress (Slot 1) =====================
    EdsUInt32 saveTarget = 1;
    err = EdsSetPropertyData(
        camera,
        kEdsPropID_Record,
        0,
        sizeof(saveTarget),
        &saveTarget);
    checkError(err, "Set Record Slot");

    // ===================== 设置容量 =====================
    EdsCapacity capacity = {0x7FFFFFFF, 0x10000, 1};
    err = EdsSetCapacity(camera, capacity);
    checkError(err, "Set Capacity");

    std::cout << "[INFO] Camera ready. Taking picture in 2 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // ===================== 拍照 =====================
    err = EdsSendCommand(camera, kEdsCameraCommand_TakePicture, 0);
    checkError(err, "Take Picture");

    std::cout << "[INFO] Waiting for image download..." << std::endl;

    // ===================== 事件循环（关键） =====================
    for (int i = 0; i < 200; ++i) {
        EdsGetEvent();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 关闭会话
    err = EdsCloseSession(camera);
    checkError(err, "Close session");

    if (camera) EdsRelease(camera);
    if (cameraList) EdsRelease(cameraList);

    // 结束 SDK
    err = EdsTerminateSDK();
    checkError(err, "Terminate SDK");

    std::cout << "[INFO] Done." << std::endl;

    return 0;
}
