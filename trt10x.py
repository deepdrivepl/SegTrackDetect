import numpy as np
import pandas as pd
import cv2
import torch
import tensorrt as trt
# import pycuda.driver as cuda
from cuda import cudart
import time



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
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        self.inputs, self.outputs, self.allocations = [], [], []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = self.context.get_tensor_shape(name)

            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_tensor_profile_shape(name, 0)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                shape = profile_shape[2]
                if self.batch_size is not None:
                    shape[0] = batch_size
                self.context.set_input_shape(name, shape)
                shape = self.context.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cudart.cudaMalloc(size)[1]
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            print(
                "{} '{}' with shape {} and dtype {}".format(
                    "Input" if is_input else "Output",
                    binding["name"],
                    binding["shape"],
                    binding["dtype"],
                )
            )

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

                
    def infer(self, batch):
        batch = np.array(batch)
        nbytes = batch.size * batch.itemsize
        cudart.cudaMemcpy(self.inputs[0]["allocation"], batch, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        self.context.execute_v2(self.allocations)
        
        for o in range(len(self.outputs)):
            nbytes = self.outputs[o]["host_allocation"].size * self.outputs[o]["host_allocation"].itemsize
            cudart.cudaMemcpy(self.outputs[o]["host_allocation"], self.outputs[o]["allocation"], nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        return [o["host_allocation"] for o in self.outputs]


def run_det_trt(engine, img):
    B,C,H,W = img.shape
    t1 = time.time()
    out = engine.infer(img)[0] # array, dtype
    t2 = time.time()
    out = out[:B,:,:]
    return torch.tensor(out), t2-t1


def run_seg_trt(engine, img):
    B,C,H,W = img.shape
    out  = engine.infer(img)[0] # array, dtype
    out = out[:B,0,...]
    return torch.tensor(out)