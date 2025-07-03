#include <iostream>
#include <vector>
#include "acl/acl.h"
// #include "aclnn_flash_attention_score.h"
#include "aclnn_wkv6.h"
#include <fstream>
#include <string>
#include <algorithm>
#include <cstdint>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

bool ReadFile(const std::string &filePath, size_t fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

int WriteDataToFile(const std::string dataPath, const std::vector<int64_t>& shape, void* deviceAddr, void** hostData) {
    auto size = GetShapeSize(shape) * sizeof(aclFloat16);    
    auto ret = aclrtMallocHost(hostData, size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*hostData, size, deviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::ofstream outFile(dataPath.c_str(), std::ios::binary);
    if (!outFile) {
        std::cerr << "Failed to open file for writing" << std::endl;
        return -1;
    }
    outFile.write(reinterpret_cast<char*>(*hostData), size);
    outFile.close();
    return 0;
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateContext(context, deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetCurrentContext(*context);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  const short batch = 1;
  const short headnum = 1;
  const short T = 64;
  const short headsize = 64;

  std::vector<int64_t> qShape = {batch, headnum, T, headsize};
  std::vector<int64_t> kShape = {batch, headnum, T, headsize};
  std::vector<int64_t> vShape = {batch, headnum, T, headsize};
  std::vector<int64_t> wShape = {batch, headnum, T, headsize};
  std::vector<int64_t> uShape = {headnum, headsize};
  std::vector<int64_t> hiShape = {batch, headnum, headsize, headsize};
  std::vector<int64_t> oShape = {batch, headnum, T, headsize};
  std::vector<int64_t> hoShape = {batch, headnum, headsize, headsize};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* wDeviceAddr = nullptr;
  void* uDeviceAddr = nullptr;
  void* hiDeviceAddr = nullptr;
  void* oDeviceAddr = nullptr;
  void* hoDeviceAddr = nullptr;


  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* w = nullptr;
  aclTensor* u = nullptr;
  aclTensor* hi = nullptr;
  aclTensor* o = nullptr;
  aclTensor* ho = nullptr;

  std::vector<aclFloat16> qHostData(batch*headnum*T*headsize);
  std::vector<aclFloat16> kHostData(batch*headnum*T*headsize);
  std::vector<aclFloat16> vHostData(batch*headnum*T*headsize);
  std::vector<aclFloat16> wHostData(batch*headnum*T*headsize);
  std::vector<aclFloat16> uHostData(headnum*headsize);
  std::vector<aclFloat16> hiHostData(batch*headnum*headsize*headsize);
  std::vector<aclFloat16> OHostData(batch*headnum*T*headsize);
  std::vector<aclFloat16> HOHostData(batch*headnum*headsize*headsize);

  size_t fileSize = 0;
  void ** input1=(void **)(&qHostData);
  void ** input2=(void **)(&kHostData);
  void ** input3=(void **)(&vHostData);
  void ** input4=(void **)(&wHostData);
  void ** input5=(void **)(&uHostData);
  void ** input6=(void **)(&hiHostData);
  size_t dataType = sizeof(aclFloat16);
  ReadFile("../input/input_q.bin", fileSize, *input1, qShape[0]*qShape[1]*qShape[2]*qShape[3]*dataType);
  ReadFile("../input/input_k.bin", fileSize, *input2, kShape[0]*kShape[1]*kShape[2]*kShape[3]*dataType);
  ReadFile("../input/input_v.bin", fileSize, *input3, vShape[0]*vShape[1]*vShape[2]*vShape[3]*dataType);
  ReadFile("../input/input_w.bin", fileSize, *input4, wShape[0]*wShape[1]*wShape[2]*wShape[3]*dataType);
  ReadFile("../input/input_u.bin", fileSize, *input5, uShape[0]*uShape[1]*dataType);
  ReadFile("../input/input_h0.bin", fileSize, *input6, hiShape[0]*hiShape[1]*hiShape[2]*hiShape[3]*dataType);
  // for (int i = 0; i < 100; ++i) {
  //     printf("qHostData[%d] = %f\n", i, aclFloat16ToFloat(qHostData[i]));
  // }
  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_FLOAT16, &w);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(uHostData, uShape, &uDeviceAddr, aclDataType::ACL_FLOAT16, &u);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(hiHostData, hiShape, &hiDeviceAddr, aclDataType::ACL_FLOAT16, &hi);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(OHostData, oShape, &oDeviceAddr, aclDataType::ACL_FLOAT16, &o);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(HOHostData, hoShape, &hoDeviceAddr, aclDataType::ACL_FLOAT16, &ho);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnwkv6GetWorkspaceSize(k, v, w, q, u, hi, o, ho, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnwkv6(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScore failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 输出o数据
std::string OPath = "../output/output_o.bin";
void* ohostData = nullptr;
WriteDataToFile(OPath, oShape, oDeviceAddr, &ohostData);
aclFloat16* o_host_ptr = static_cast<aclFloat16*>(ohostData);
for (int i = 0; i < 100; ++i) {
    printf("ohostData[%d] = %f\n", i, aclFloat16ToFloat(o_host_ptr[i]));
}

// 输出ho数据
std::string HOPath = "../output/output_ho.bin";
void* hohostData = nullptr;
WriteDataToFile(HOPath, hoShape, hoDeviceAddr, &hohostData);
aclFloat16* ho_host_ptr = static_cast<aclFloat16*>(hohostData);
for (int i = 0; i < 100; ++i) {
    printf("hohostData[%d] = %f\n", i, aclFloat16ToFloat(ho_host_ptr[i]));
}

 
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(w);
  aclDestroyTensor(u);
  aclDestroyTensor(hi);
  aclDestroyTensor(o);
  aclDestroyTensor(ho);
  
  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(wDeviceAddr);
  aclrtFree(uDeviceAddr);
  aclrtFree(hiDeviceAddr);
  aclrtFree(oDeviceAddr);
  aclrtFree(hoDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();
  
  return 0;
}
