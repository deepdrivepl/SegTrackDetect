import numpy as np
import pandas as pd
import cv2
import torch
import tensorrt as trt
import time
import pycuda.driver as cuda



## ===== TensorRT Engine =====
class BaseEngine(object):
    def __init__(self, img_size, engine_path, names, batch_size=None):

        self.names = names
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.batch_size = batch_size
        trt.init_libnvinfer_plugins(self.logger,'')
        runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # fixes PyCUDA ERROR: The context stack was not empty upon module cleanup.
        with self.engine.create_execution_context() as context:
            ctx = cuda.Context.attach()
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings = [], [], []
            self.stream = cuda.Stream()
        
            for binding in self.engine:
                shape = self.engine.get_binding_shape(binding)
                if shape[0] < 0:
                    shape[0] = self.batch_size
                
                size = trt.volume(shape)
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings.append(int(device_mem))
                if self.engine.binding_is_input(binding):
                    self.inputs.append({'host': host_mem, 'device': device_mem})
                else:
                    self.outputs.append({'host': host_mem, 'device': device_mem})
            ctx.detach()


                
    def infer(self, batch):
        batch = np.array(batch)
        self.context.set_binding_shape(0, (batch.shape))
        self.inputs[0]['host'] = batch

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        return [out['host'] for out in self.outputs]


def run_det_trt(engine, img):
    B,C,H,W = img.shape
    t1 = time.time()
    out = engine.infer(img)[0] # array, dtype
    t2 = time.time()
    out = np.reshape(out, (1,-1,9))
    return torch.tensor(out), t2-t1


def run_seg_trt(engine, img):
    B,C,H,W = img.shape
    t1 = time.time()
    out  = engine.infer(img)[0] # array, dtype
    t2 = time.time()
    out = np.reshape(out, (-1,H,W))
    out = torch.tensor(out[0,...])
    return out, t2-t1