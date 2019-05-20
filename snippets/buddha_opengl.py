from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

a_np = np.random.rand(50).astype(np.float32)
b_np = np.random.rand(50).astype(np.float32)
print("arrays created")
# ctx = cl.create_some_context()
playforms = cl.get_platforms()
print("platforms gotten")
quit()
ctx = cl.Context(
	dev_type = cl.device_type.ALL,
	properties = [(cl.context_properties.PLATFORM, platforms[0])]
)
print("context created")
queue = cl.CommandQueue(ctx)
print("queue created")

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
print("buffers created")

prg = cl.Program(ctx, """
__kernel void sum(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
{
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
""").build()
print("built")
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
print("summed")
res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)
print("enqueued")
# Check on CPU with Numpy:
print(res_np - (a_np + b_np))
print(np.linalg.norm(res_np - (a_np + b_np)))