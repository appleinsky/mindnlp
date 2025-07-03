#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;


class Kernelwkv6
{
public:
    __aicore__ inline Kernelwkv6() {}
    __aicore__ inline void Init(uint32_t tileNum, uint32_t tileNumRemainLength, uint32_t totalHeads, uint32_t T,
                                uint32_t tileLength, uint32_t HEAD_SIZE, uint32_t HEAD_NUMS, bool hasRemainer,
                                GM_ADDR k, GM_ADDR v, GM_ADDR w, GM_ADDR r, 
                                GM_ADDR u, GM_ADDR h0, GM_ADDR o,  GM_ADDR ht)
    {
        this->tileNum = tileNum;
        this->tileNumRemainLength = tileNumRemainLength;
        this->totalHeads = totalHeads;
        this->T = T;
        this->tileLength = tileLength;
        this->HEAD_SIZE = HEAD_SIZE;
        this->HEAD_NUMS = HEAD_NUMS;
        this->HEAD_ELEMENTS = this->HEAD_SIZE * this->HEAD_SIZE;
        this->scale = 0.5;
        this->hasRemainer = hasRemainer;
        uint32_t blockNum = GetBlockNum();
        uint32_t baseHeadsPerCore = totalHeads / blockNum; // 基础分配
        uint32_t remainerHeads = totalHeads % blockNum;   // 余数
        uint32_t currentBlock = GetBlockIdx();
        // 计算当前核心实际处理的head数量
        this->headPerCore = baseHeadsPerCore;
        if (currentBlock < remainerHeads)
        {
            this->headPerCore += 1; // 前面几个核多处理一个head
        }
        // 计算当前核心的数据偏移
        uint32_t headOffset = baseHeadsPerCore * currentBlock;
        if (currentBlock < remainerHeads)
        {
            headOffset += currentBlock;
        }
        else
        {
            headOffset += remainerHeads;
        }
        uint32_t uh_offset = headOffset % this->HEAD_NUMS;
        this->sizePerCore = this->headPerCore * this->T * this->HEAD_SIZE;
        kGm.SetGlobalBuffer((__gm__ float *)k + headOffset * this->T * this->HEAD_SIZE, this->sizePerCore);
        vGm.SetGlobalBuffer((__gm__ float *)v + headOffset * this->T * this->HEAD_SIZE, this->sizePerCore);
        wGm.SetGlobalBuffer((__gm__ float *)w + headOffset * this->T * this->HEAD_SIZE, this->sizePerCore);
        rGm.SetGlobalBuffer((__gm__ float *)r + headOffset * this->T * this->HEAD_SIZE, this->sizePerCore);
        oGm.SetGlobalBuffer((__gm__ float *)o + headOffset * this->T * this->HEAD_SIZE, this->sizePerCore);
        uGm.SetGlobalBuffer((__gm__ float *)u + uh_offset * this->HEAD_SIZE, this->headPerCore * this->HEAD_SIZE);
        h0Gm.SetGlobalBuffer((__gm__ float *)h0 + headOffset * this->HEAD_ELEMENTS, this->headPerCore * this->HEAD_ELEMENTS);
        htGm.SetGlobalBuffer((__gm__ float *)ht + headOffset * this->HEAD_ELEMENTS, this->headPerCore * this->HEAD_ELEMENTS);
        // k,v,w,r,u,o每次搬运[tileLength, N]大小的tensor
        pipe.InitBuffer(inQueueK, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float));
        pipe.InitBuffer(inQueueV, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float));
        pipe.InitBuffer(inQueueW, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float));
        pipe.InitBuffer(inQueueR, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float));
        pipe.InitBuffer(inQueueU, BUFFER_NUM, this->HEAD_SIZE * sizeof(float));
        // h0, ht每次搬运[HEAD_SIZE, HEAD_SIZE]大小的tensor
        pipe.InitBuffer(inQueueH, BUFFER_NUM, this->HEAD_ELEMENTS * sizeof(float));
        pipe.InitBuffer(outQueueH, BUFFER_NUM, this->HEAD_ELEMENTS * sizeof(float));
        // 其中 o 既是输入也是输出，所以既需要vecin的buffer也需要vecout的buffer
        pipe.InitBuffer(inQueueO, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float));
        pipe.InitBuffer(outQueueO, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float));
        // state及中间变量，每个中间变量大小为[N, N]
        pipe.InitBuffer(stateBuf, 3 * this->HEAD_ELEMENTS * sizeof(float));
        // 用于储存broadcast结果
        pipe.InitBuffer(broadBuf0, this->HEAD_ELEMENTS * sizeof(float));
        pipe.InitBuffer(broadBuf1, this->HEAD_ELEMENTS * sizeof(float));
        pipe.InitBuffer(broadBuf2, this->HEAD_ELEMENTS * sizeof(float));
        // 设置broadcast shape参数
        SetBroadShapes();
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> stateLocal = stateBuf.Get<float>();
        LocalTensor<float> broadLocal0 = broadBuf0.Get<float>();
        LocalTensor<float> broadLocal1 = broadBuf1.Get<float>();
        LocalTensor<float> broadLocal2 = broadBuf2.Get<float>();


        for (uint32_t h = 0; h < this->headPerCore; h++)
        {
            // copy tensor u[h,:]
            CopyInU(h);
            LocalTensor<float> uLocal = inQueueU.DeQue<float>();
            // broadcast u and store in broadLocal0:[1, N] to [N, N]
            BroadCast<float, 2, 0>(broadLocal0, uLocal, broadDstShape, broadSrcShape);

            // 加载当前头的初始 h0 到 stateLocal[0]
            uint32_t h_offset = h * this->HEAD_ELEMENTS;
            CopyInH0(h, h_offset);
            LocalTensor<float> hLocal = inQueueH.DeQue<float>();
            DataCopy(stateLocal[0], hLocal, this->HEAD_ELEMENTS);
            // for (int32_t i = 0; i < this->HEAD_ELEMENTS; ++i) {
            //     printf("h0[%d]: %f ", i, float16_to_float(hLocal[i])); 
            
            // }
            inQueueH.FreeTensor(hLocal);
            
            printf("head_elements nums %d\n", this->HEAD_ELEMENTS);

            for (uint32_t tile = 0; tile < this->tileNum; tile++)
            {
                // copy tensor k,v,w,r,o[b, h, tile * tileLength:(tile+1)*tileLength, :]
                CopyInKVWRO(h, tile, this->tileLength);
                LocalTensor<float> kLocal = inQueueK.DeQue<float>();
                LocalTensor<float> vLocal = inQueueV.DeQue<float>();
                LocalTensor<float> wLocal = inQueueW.DeQue<float>();
                LocalTensor<float> rLocal = inQueueR.DeQue<float>();
                LocalTensor<float> oLocal = inQueueO.DeQue<float>();
                Compute(kLocal, vLocal, wLocal, rLocal, oLocal, stateLocal, broadLocal0, broadLocal1, broadLocal2, h, 
                        tile, this->tileLength);
                CopyOutO(h, tile, this->tileLength);
                
            }

            // 处理余数
            if (this->hasRemainer)
            {
                CopyInKVWRO(h, this->tileNum, this->tileNumRemainLength);
                LocalTensor<float> kLocal = inQueueK.DeQue<float>();
                LocalTensor<float> vLocal = inQueueV.DeQue<float>();
                LocalTensor<float> wLocal = inQueueW.DeQue<float>();
                LocalTensor<float> rLocal = inQueueR.DeQue<float>();
                LocalTensor<float> oLocal = inQueueO.DeQue<float>();
                
                Compute(kLocal, vLocal, wLocal, rLocal, oLocal, stateLocal, broadLocal0, broadLocal1, broadLocal2, h, 
                        this->tileNum, this->tileNumRemainLength);
                CopyOutO(h, this->tileNum, this->tileNumRemainLength);
            }

            // copy out stateLocal[0] to ht[b, h, :, :]
            LocalTensor<float> htOutLocal = outQueueH.AllocTensor<float>();
            DataCopy(htOutLocal, stateLocal[0], this->HEAD_ELEMENTS);
            DumpTensor(htOutLocal, 11, 4096); 
            outQueueH.EnQue<float>(htOutLocal);      
            CopyOutHt(h, h_offset);
            
            inQueueU.FreeTensor(uLocal);
        }
    }

private:
    // inline void printfloat_16(const float fnum){
    //         uint8_t* c = (uint8_t*)&fnum;   //指针指向低位字节的地址，所以最先传给c的是低位字节
    //             printf("0x");
    //         for(auto i = 3;i >= 0;i--)
    //             printf("%02x",c[i]);       //先获得的是低位字节，所以应该把低位字节放在最后面
    // }

    __aicore__ inline void SetBroadShapes()
    {
        // 除了v以外所有的broadcast shape: [1, N] -> [N, N]
        broadDstShape[0] = this->HEAD_SIZE;
        broadDstShape[1] = this->HEAD_SIZE;
        broadSrcShape[0] = 1;
        broadSrcShape[1] = this->HEAD_SIZE;
        // v的broadcast shape: [N, 1] -> [N, N]
        vDstShape[0] = this->HEAD_SIZE;
        vDstShape[1] = this->HEAD_SIZE;
        vSrcShape[0] = this->HEAD_SIZE;
        vSrcShape[1] = 1;
    }

    __aicore__ inline void CopyInU(uint32_t progress_h)
    {
        // copy in u[h,:]
        uint32_t offset = progress_h * this->HEAD_SIZE;
        LocalTensor<float> uLocal = inQueueU.AllocTensor<float>();
        DataCopy(uLocal, uGm[offset], this->HEAD_SIZE);
        inQueueU.EnQue<float>(uLocal);
    }

    __aicore__ inline void CopyInKVWRO(uint32_t progress_h, uint32_t progress_tile, uint32_t currentTileLength)
    {
        // copy k,v,w,r,o[b, h, tile*tileLength:(tile+1)*tileLength, :]
        
        uint32_t offset = progress_h * this->T * this->HEAD_SIZE + progress_tile * this->tileLength * this->HEAD_SIZE;
        LocalTensor<float> kLocal = inQueueK.AllocTensor<float>();
        LocalTensor<float> vLocal = inQueueV.AllocTensor<float>();
        LocalTensor<float> wLocal = inQueueW.AllocTensor<float>();
        LocalTensor<float> rLocal = inQueueR.AllocTensor<float>();
        LocalTensor<float> oLocal = inQueueO.AllocTensor<float>();
        DataCopy(kLocal, kGm[offset], currentTileLength * this->HEAD_SIZE);
        DataCopy(vLocal, vGm[offset], currentTileLength * this->HEAD_SIZE);
        DataCopy(wLocal, wGm[offset], currentTileLength * this->HEAD_SIZE);
        DataCopy(rLocal, rGm[offset], currentTileLength * this->HEAD_SIZE);
        DataCopy(oLocal, oGm[offset], currentTileLength * this->HEAD_SIZE);
        inQueueK.EnQue<float>(kLocal);
        inQueueV.EnQue<float>(vLocal);
        inQueueW.EnQue<float>(wLocal);
        inQueueR.EnQue<float>(rLocal);
        inQueueO.EnQue<float>(oLocal);
    }

    __aicore__ inline void CopyInH0(uint32_t progress_h, uint32_t offset)
    {
        // copy in h0[b, h, HEAD_SIZE, HEAD_SIZE]
        LocalTensor<float> hLocal = inQueueH.AllocTensor<float>();
        DataCopy(hLocal, h0Gm[offset], this->HEAD_ELEMENTS);
        inQueueH.EnQue<float>(hLocal);
    }

    __aicore__ inline void CopyOutO(uint32_t progress_h, uint32_t progress_tile, uint32_t currentTileLength)
    {
        // copy out o[b, h, tile*tileLength:(tile+1)*tileLength,:]
        uint32_t offset = progress_h * this->T * this->HEAD_SIZE + progress_tile * this->tileLength * this->HEAD_SIZE;
        LocalTensor<float> oOutLocal = outQueueO.DeQue<float>();
        // DumpTensor(oOutLocal, 11, 4096); 
        DataCopy(oGm[offset], oOutLocal, currentTileLength * this->HEAD_SIZE);
        // DumpTensor(oGm[offset], 12, 4096); 
        outQueueO.FreeTensor(oOutLocal);
    }

    __aicore__ inline void CopyOutHt(uint32_t progress_h, uint32_t offset)
    {
        LocalTensor<float> htOutLocal = outQueueH.DeQue<float>();
        DataCopy(htGm[offset], htOutLocal, this->HEAD_ELEMENTS); 
        outQueueH.FreeTensor(htOutLocal);       
    }

    __aicore__ inline void Compute(LocalTensor<float> kLocal, LocalTensor<float> vLocal, LocalTensor<float> wLocal,
                                   LocalTensor<float> rLocal, LocalTensor<float> oLocal, LocalTensor<float> stateLocal,
                                   LocalTensor<float> broadLocal0, LocalTensor<float> broadLocal1, 
                                   LocalTensor<float> broadLocal2,
                                   uint32_t progress_h, uint32_t progress_tile,
                                   uint32_t currentTileLength)
    {
        uint32_t offset0 = 0; // reserved for state vectors
        uint32_t offset1 = this->HEAD_ELEMENTS;
        uint32_t offset2 = this->HEAD_ELEMENTS * 2;

        // DumpTensor(stateLocal, 8, 4096);
        for (uint32_t t = 0; t < currentTileLength; t++)
        {
            // compute kv = k.mT@v, offset1
            // broadcast v from [N,1] to [N, N]
            Muls(vLocal[t * this->HEAD_SIZE], vLocal[t * this->HEAD_SIZE], this->scale, this->HEAD_SIZE);
            PipeBarrier<PIPE_V>();
            BroadCast<float, 2, 1>(broadLocal2, vLocal[t * this->HEAD_SIZE], vDstShape, vSrcShape);
            // broadcast k from [1,N] to [N, N]
            // DumpTensor(vLocal[t * this->HEAD_SIZE], 0, this->HEAD_SIZE);
            Muls(kLocal[t * this->HEAD_SIZE], kLocal[t * this->HEAD_SIZE], this->scale, this->HEAD_SIZE);
            PipeBarrier<PIPE_V>();
            // DumpTensor(kLocal[t * this->HEAD_SIZE], 1, this->HEAD_SIZE);
            BroadCast<float, 2, 0>(broadLocal1, kLocal[t * this->HEAD_SIZE], broadDstShape, broadSrcShape);
            PipeBarrier<PIPE_V>();
            // DumpTensor(kLocal[t * this->HEAD_SIZE], 2, this->HEAD_SIZE);
            Mul(stateLocal[offset1], broadLocal1, broadLocal2, this->HEAD_ELEMENTS);

            
            // DumpTensor(stateLocal[offset1], 3, this->HEAD_SIZE);
            PipeBarrier<PIPE_V>();

            // compute ukv = u * kv, shape: N * N, offset2, u was stored in broadLocal0
            Mul(stateLocal[offset2], broadLocal0, stateLocal[offset1], this->HEAD_ELEMENTS);
            // printf("ukv: %f \n", float16_to_float(stateLocal[offset2]));
            // DumpTensor(stateLocal[offset2], 4, this->HEAD_SIZE);
            PipeBarrier<PIPE_V>();

            // compute sukv = state + ukv, shape:N * N, offset2
            Add(stateLocal[offset2], stateLocal[offset2], stateLocal[offset0], this->HEAD_ELEMENTS);
            // DumpTensor(stateLocal[offset2], 5, this->HEAD_SIZE);
            // printf("sukv: %f \n", float16_to_float(stateLocal[offset2]));
            PipeBarrier<PIPE_V>();

            // compute state = exp(-exp(w)) * state, shape:N * N, state
            // broadcast w from [1, N] to [N, N]
            Exp(wLocal[t * this->HEAD_SIZE], wLocal[t * this->HEAD_SIZE], this->HEAD_SIZE);
            PipeBarrier<PIPE_V>();
            Muls(wLocal[t * this->HEAD_SIZE], wLocal[t * this->HEAD_SIZE], (float)-1.0, this->HEAD_SIZE);
            PipeBarrier<PIPE_V>();
            Exp(wLocal[t * this->HEAD_SIZE], wLocal[t * this->HEAD_SIZE], this->HEAD_SIZE);
            PipeBarrier<PIPE_V>();
            BroadCast<float, 2, 0>(broadLocal1, wLocal[t * this->HEAD_SIZE], broadDstShape, broadSrcShape);
            PipeBarrier<PIPE_V>();
            Mul(stateLocal[offset0], broadLocal1, stateLocal[offset0], this->HEAD_ELEMENTS);
            // DumpTensor(stateLocal[offset0], 6, this->HEAD_SIZE);
            // printf("state_w: %f \n", float16_to_float(stateLocal[offset0]));
            PipeBarrier<PIPE_V>();

            // compute state = state + kv, shape:N*N, state
            Add(stateLocal[offset0], stateLocal[offset0], stateLocal[offset1], this->HEAD_ELEMENTS);
            // DumpTensor(stateLocal[offset0], 7, this->HEAD_SIZE);
            // printf("state: %f \n", float16_to_float(stateLocal[offset0]));
            // compute out = r * sukv, shape:N * N, offset2
            // broadcast r from [1, N] to [N, N]
            BroadCast<float, 2, 0>(broadLocal1, rLocal[t * this->HEAD_SIZE], broadDstShape, broadSrcShape);
            PipeBarrier<PIPE_V>();
            Mul(stateLocal[offset2], broadLocal1, stateLocal[offset2], this->HEAD_ELEMENTS);
            PipeBarrier<PIPE_V>();
            // printf("out: %f \n", stateLocal[offset0]);
            // // DumpTensor(stateLocal, 8, 32); 
            

            // compute reduceSum(out), shape: N
            // mask=N, repeatTimes=N, dstRepStride=1, srcBlkStride=1, srcRepStride=N*sizeof(float)/32=4
            WholeReduceSum(oLocal[t * this->HEAD_SIZE], stateLocal[offset2], this->HEAD_SIZE,
                                this->HEAD_SIZE, 1, 1, this->HEAD_SIZE * sizeof(float) / 32);

            // DumpTensor(oLocal[t * this->HEAD_SIZE], 9, this->HEAD_SIZE); 
        }

        // move o from vecin to vecout then free vecin o
        LocalTensor<float> oOutLocal = outQueueO.AllocTensor<float>();
        DataCopy(oOutLocal, oLocal, this->tileLength * this->HEAD_SIZE);
        DumpTensor(oOutLocal, 10, 4096); 
        outQueueO.EnQue<float>(oOutLocal);

        inQueueO.FreeTensor(oLocal);

        // free k,v,w,r vecin for reuse
        inQueueK.FreeTensor(kLocal);
        inQueueV.FreeTensor(vLocal);
        inQueueW.FreeTensor(wLocal);
        inQueueR.FreeTensor(rLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueK, inQueueV, inQueueW, 
                inQueueR, inQueueU, inQueueO, inQueueH;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueO, outQueueH;
    GlobalTensor<float> kGm, vGm, wGm, rGm, uGm, oGm, h0Gm, htGm;
    TBuf<QuePosition::VECCALC> stateBuf, broadBuf0, broadBuf1, broadBuf2;
    uint32_t  tileLength, HEAD_NUMS, HEAD_SIZE, HEAD_ELEMENTS;
    uint32_t tileNum, tileNumRemainLength, baseHeadsPerCore, remainerHeads, totalHeads;
    uint32_t batchPerCore, sizePerCore, headPerCore, uSizePerCore;
    uint32_t T;
    float scale;
    bool hasRemainer;
    uint32_t broadDstShape[2], broadSrcShape[2];
    uint32_t vDstShape[2], vSrcShape[2];
};


// implementation of kernel function
extern "C" __global__ __aicore__ void wkv6(GM_ADDR k, GM_ADDR v, GM_ADDR w, GM_ADDR r,
                                                     GM_ADDR u, GM_ADDR h0, GM_ADDR o, GM_ADDR ht, 
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 使能VectorCore
    Kernelwkv6 op;
    op.Init(tiling_data.tileNum, tiling_data.tileNumRemainLength, tiling_data.totalHeads, tiling_data.T,
            tiling_data.tileLength, tiling_data.HEAD_SIZE, tiling_data.HEAD_NUMS, tiling_data.hasRemainer,
            k, v, w, r, u, h0, o, ht);
    op.Process();
}