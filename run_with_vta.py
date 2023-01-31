from __future__ import absolute_import, print_function
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 17:20:32 2023

@author: libo
"""

import os, time
import numpy as np
import tvm
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_executor, utils
import vta
from define_model import *
import torch
assert tvm.runtime.enabled("rpc")

#define model and platform
env = vta.get_env()
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

pack_dict = {
    "MyMode":["nn.dense", "sigmoid"],
}

model = "MyMode"
assert model in pack_dict

# set vta to run the model
if env.TARGET not in ["sim", "tsim", "intelfocl"]:
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = os.environ.get("TVM_TRACKER_PORT", None)
    device_host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")
    device_port = os.environ.get("VTA_RPC_PORT", "9091")
    if not tracker_host or not tracker_port:
        remote = rpc.connect(device_host, int(device_port))
    else:
        remote = autotvm.measure.request_remote(
            env.TARGET, tracker_host, int(tracker_port), timeout=10000
        )

    reconfig_start = time.time()
    vta.reconfig_runtime(remote)
    vta.program_fpga(remote, bitstream="/home/libo/tvmTE0802/vta/tutorials/vta.bit")
    reconfig_time = time.time() - reconfig_start
    print("Reconfigured FPGA and RPC runtime in {0:.2f}s!".format(reconfig_time))

else:
    remote = rpc.LocalSession()

    if env.TARGET in ["intelfocl"]:
        # program intelfocl aocx
        vta.program_fpga(remote, bitstream="vta.bitstream")

# Get execution context from remote
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)

# set model and input
with autotvm.tophub.context(target):
    
    
    gluon_model = torch.load("MyMod.pth")
    
    ###
    # hyperparameters

    #original input size
    # input_shape = [1, 44]   
    # my_input_data = get_test_data().reshape(input_shape)

    # skip L1
    # input_shape = [1, 24]   
    # my_input_data = np.load("x1.npy").reshape(input_shape)
    # gluon_model.bypass_x1 = True

    # skip L12
    # input_shape = [1, 10]   
    # my_input_data = np.load("x2.npy").reshape(input_shape)
    # gluon_model.bypass_x1 = True
    # gluon_model.bypass_x2 = True

    # skip L123
    input_shape = [1, 1]   
    my_input_data = np.load("x3.npy").reshape(input_shape)
    gluon_model.bypass_x1 = True
    gluon_model.bypass_x2 = True
    gluon_model.bypass_x3 = True
    ###
    
    # print orignal output
    print("original output: ", gluon_model(torch.tensor(my_input_data)))
    
    # set build start time
    build_start = time.time()
    
    #this is initial input, only for the form, model don't use this input
    input_data = torch.zeros(input_shape)
    scripted_model = torch.jit.trace(gluon_model, input_data).eval()
    
    input_name = "input0"
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    
    # vta optimize model
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params=params)
        # Perform graph packing and constant folding for VTA target
        assert env.BLOCK_IN == env.BLOCK_OUT

        relay_prog = mod["main"]
    
    if env.TARGET == "intelfocl":
        # multiple targets to run both on cpu and vta
        target = {"cpu": env.target_vta_cpu, "ext_dev": target}
    with vta.build_config(
        opt_level=3, disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}
    ):
        graph, lib, params = relay.build(
            relay_prog, target=tvm.target.Target(target, host=env.target_host), params=params
        )


    # compute Relay build time
    build_time = time.time() - build_start
    print(model + " inference graph built in {0:.2f}s!".format(build_time))

    # Send the inference library over to the remote RPC server
    temp = utils.tempdir()
    lib.export_library(temp.relpath("graphlib.tar"))
    remote.upload(temp.relpath("graphlib.tar"))
    lib = remote.load_module("graphlib.tar")

    if env.TARGET == "intelfocl":
        ctxes = [remote.ext_dev(0), remote.cpu(0)]
        m = graph_executor.create(graph, lib, ctxes)
    else:
        m = graph_executor.create(graph, lib, ctx)


# Set inputs and parameters
dtype = "float32"
m.set_input(**params)
m.set_input(input_name, my_input_data.astype(dtype))

# Perform inference and gather execution statistics
num = 4  # number of times we run model for a single measurement
rep = 3  # number of measurements
timer = m.module.time_evaluator("run", ctx, number=num, repeat=rep)

tcost = timer()
mean = tcost.mean * 1000000
print("Average per sample inference time: %.2fus" % (mean))
print("max={:.2f}[us], min={:.2f}[us], median={:.2f}[us], mean={:.2f}[us], std={:.2f}[us]".
      format(tcost.max*1000000, tcost.min*1000000, tcost.median*1000000, tcost.mean*1000000, tcost.std*1000000))

# get vta's output, compare with original output
vta_output = m.get_output(0)
print("vta output: ", vta_output)
