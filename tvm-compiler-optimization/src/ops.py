import tvm
from tvm import te
import numpy as np

def make_conv1d_cpu_scheduler_baseline(M, N):
    """
    Baseline implementation of 1D convolution for CPU.
    This is a simple, direct implementation without optimizations,
    which should work correctly for all input sizes.
    
    Args:
        M: size of the first input array
        N: size of the second input array
        
    Returns:
        s: schedule
        A: first input placeholder
        W: second input placeholder (weights)
        B: output tensor
    """
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")

    k = te.reduce_axis((0, M), "k")
    
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(
            tvm.tir.if_then_else(
                tvm.tir.all(k >= 0, k < M, n - k >= 0, n - k < N),
                A[k] * W[n - k],
                tvm.tir.const(0.0, "float32")
            ),
            axis=k
        ),
        name="B"
    )
    
    s = te.create_schedule(B.op)
    return s, A, W, B


def make_conv1d_cpu_scheduler_with_tiling(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")
    k = te.reduce_axis((0, M), "k")
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(tvm.tir.if_then_else(
            tvm.tir.any(k < 0, k >= M, n - k < 0, n - k >= N),
            tvm.tir.const(0.0, "float32"),
            A[k] * W[n - k]), axis=k),
        name="B",
    )
    s = te.create_schedule(B.op)
    
    if M <= 2 or N <= 2:
        return s, A, W, B

    output_axis = B.op.axis[0]
    tile_size = min(32, max(4, (M + N - 1) // 16))
    
    if M + N - 1 > tile_size * 2:
        outer, inner = s[B].split(output_axis, factor=tile_size)
        
        ko, ki = s[B].split(k, factor=tile_size)
        
        s[B].reorder(outer, ko, inner, ki)
    
    return s, A, W, B


def make_conv1d_cpu_scheduler_with_parallel(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")
    k = te.reduce_axis((0, M), "k")
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(tvm.tir.if_then_else(
            tvm.tir.any(k < 0, k >= M, n - k < 0, n - k >= N),
            tvm.tir.const(0.0, "float32"),
            A[k] * W[n - k]), axis=k),
        name="B",
    )
    s = te.create_schedule(B.op)

    if M + N <= 128:
        return s, A, W, B
    output_axis = B.op.axis[0]
    chunk_size = min(64, max(8, (M + N - 1) // 8))
    
    if M + N - 1 > chunk_size * 2:
        outer, inner = s[B].split(output_axis, factor=chunk_size)
        s[B].parallel(outer)
    
    return s, A, W, B


def make_conv1d_cpu_scheduler_with_vectorization(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")
    k = te.reduce_axis((0, M), "k")
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(tvm.tir.if_then_else(
            tvm.tir.any(k < 0, k >= M, n - k < 0, n - k >= N),
            tvm.tir.const(0.0, "float32"),
            A[k] * W[n - k]), axis=k),
        name="B",
    )
    s = te.create_schedule(B.op)
    
    if min(M, N) <= 8:
        return s, A, W, B
    output_axis = B.op.axis[0]
    vec_len = 8 
    
    if M + N - 1 > vec_len * 2:
        outer, inner = s[B].split(output_axis, factor=vec_len)
        
        if M + N - 1 > vec_len * 4:
            ko, ki = s[B].split(k, factor=vec_len * 2)
            s[B].reorder(outer, ko, inner, ki)
        
        s[B].vectorize(inner)
    
    return s, A, W, B


def make_conv1d_cpu_scheduler_optimized(M, N, use_tiling=True, use_parallel=True, use_vectorization=True):
    """
    Implement 1D convolution with combined optimizations.
    
    Args:
        M: size of the first input array
        N: size of the second input array
        use_tiling: whether to use tiling optimization
        use_parallel: whether to use parallelization
        use_vectorization: whether to use vectorization
        
    Returns:
        s: schedule
        A: first input placeholder
        W: second input placeholder (weights)
        B: output tensor
    """
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")
    
    output_len = M + N - 1
    is_large_input = (M > 500 or N > 500)
    

    if is_large_input:
        k = te.reduce_axis((0, M), "k")
        
        B = te.compute(
            (output_len,),
            lambda n: te.sum(
                tvm.tir.if_then_else(
                    tvm.tir.all(k >= 0, k < M, n - k >= 0, n - k < N),
                    A[k] * W[n - k],
                    tvm.tir.const(0.0, "float32")
                ),
                axis=k
            ),
            name="B"
        )
    else:
        k = te.reduce_axis((0, M), "k")
        
        B = te.compute(
            (output_len,),
            lambda n: te.sum(
                tvm.tir.if_then_else(
                    tvm.tir.all(k >= 0, k < M, n - k >= 0, n - k < N),
                    A[k] * W[n - k],
                    tvm.tir.const(0.0, "float32")
                ),
                axis=k
            ),
            name="B"
        )
    
    s = te.create_schedule(B.op)
    
    if M <= 8 or N <= 8:
        return s, A, W, B
    
    output_axis = B.op.axis[0]
    
    if is_large_input:
        chunk_size = min(128, max(16, output_len // 32))
        
        outer, inner = s[B].split(output_axis, factor=chunk_size)
        
        if use_parallel:
            s[B].parallel(outer)

        if M > 128:
            ko, ki = s[B].split(k, factor=64)
            s[B].reorder(outer, ko, inner, ki)
        else:
            s[B].reorder(outer, inner, k)
            
        if use_vectorization and output_len > 1024 and chunk_size > 8:
            inner_outer, inner_inner = s[B].split(inner, factor=4)
            s[B].vectorize(inner_inner)
            
        return s, A, W, B
    

    if use_tiling:
        output_tile_factor = min(32, max(4, output_len // 8))
        reduce_tile_factor = min(32, max(4, M // 8))
        
        if output_len > output_tile_factor * 2:
            outer, inner = s[B].split(output_axis, factor=output_tile_factor)
            
            if M > reduce_tile_factor * 2:
                ko, ki = s[B].split(k, factor=reduce_tile_factor)
                s[B].reorder(outer, ko, inner, ki)
            else:
                s[B].reorder(outer, inner, k)
    
    if use_parallel and output_len >= 64:
        if use_tiling and 'outer' in locals():
            s[B].parallel(outer)
        else:
            chunk_size = min(32, max(4, output_len // 8))
            if output_len > chunk_size * 2:
                outer, inner = s[B].split(output_axis, factor=chunk_size)
                s[B].parallel(outer)
    
    if use_vectorization and output_len >= 32:
        vec_factor = 4 
            
        if use_tiling and 'inner' in locals():
            if output_tile_factor >= vec_factor * 2:
                inner_outer, inner_inner = s[B].split(inner, factor=vec_factor)
                s[B].vectorize(inner_inner)
        elif not use_tiling and not use_parallel:
            if output_len > vec_factor * 2:
                outer_vec, inner_vec = s[B].split(output_axis, factor=vec_factor)
                s[B].vectorize(inner_vec)
    
    return s, A, W, B


def make_conv1d_cpu_scheduler(M, N):
    """    
    Args:
        M: size of the first input array
        N: size of the second input array
        
    Returns:
        s: schedule
        A: first input placeholder
        W: second input placeholder (weights)
        B: output tensor
    """
    if M <= 2 or N <= 2:
        return make_conv1d_cpu_scheduler_baseline(M, N)
    
    if M >= 1000 or N >= 1000:
        if max(M, N) >= 10000:
            return make_conv1d_cpu_scheduler_optimized(M, N, 
                                               use_tiling=True, 
                                               use_parallel=True, 
                                               use_vectorization=False)
        else:
            return make_conv1d_cpu_scheduler_optimized(M, N, 
                                               use_tiling=True, 
                                               use_parallel=True, 
                                               use_vectorization=True)
    elif M <= 32 or N <= 32:
        return make_conv1d_cpu_scheduler_optimized(M, N, 
                                             use_tiling=True, 
                                             use_parallel=False, 
                                             use_vectorization=False)
    else:
        return make_conv1d_cpu_scheduler_optimized(M, N, 
                                             use_tiling=True, 
                                             use_parallel=True, 
                                             use_vectorization=True)
    

def make_conv1d_gpu_scheduler_baseline(M, N):
    """    
    Args:
        M: size of the first input array
        N: size of the second input array
        
    Returns:
        s: schedule
        A: first input placeholder
        W: second input placeholder (weights)
        B: output tensor
    """
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")
    
    k = te.reduce_axis((0, M), name="k")
    
    B = te.compute(
        (M + N - 1,),
        lambda i: te.sum(
            tvm.tir.if_then_else(
                tvm.tir.any(i - k < 0, i - k >= N),
                0.0,
                A[k] * W[i - k]
            ),
            axis=k
        ),
        name="B"
    )
    
    s = te.create_schedule(B.op)
    
    i = B.op.axis[0]
    
    block_i, thread_i = s[B].split(i, factor=64)
    
    s[B].bind(block_i, te.thread_axis("blockIdx.x"))
    s[B].bind(thread_i, te.thread_axis("threadIdx.x"))
    
    return s, A, W, B

# Optimizaiton 1
def make_optimized_memory_conv1d_gpu(M, N):
    """    
    Args:
        M: size of the first input array
        N: size of the second input array
        
    Returns:
        s: schedule
        A: first input placeholder
        W: second input placeholder (weights)
        B: output tensor
    """
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")
    
    k = te.reduce_axis((0, M), name="k")
    
    B = te.compute(
        (M + N - 1,),
        lambda i: te.sum(
            tvm.tir.if_then_else(
                tvm.tir.any(i - k < 0, i - k >= N),
                0.0,
                A[k] * W[i - k]
            ),
            axis=k
        ),
        name="B"
    )
    
    s = te.create_schedule(B.op)
    
    i = B.op.axis[0]
    
    BL = s.cache_write(B, "local")
    
    block_i, thread_i = s[B].split(i, factor=64)
    
    s[B].bind(block_i, te.thread_axis("blockIdx.x"))
    s[B].bind(thread_i, te.thread_axis("threadIdx.x"))
    
    s[BL].compute_at(s[B], thread_i)
    
    return s, A, W, B

# shared memory optimization
def make_shared_memory(M, N):
    """
    
    Args:
        M: size of the first input array
        N: size of the second input array
        
    Returns:
        s: schedule
        A: first input placeholder
        W: second input placeholder (weights)
        B: output tensor
    """
    A = te.placeholder((M,), name="A", dtype="float32")
    W = te.placeholder((N,), name="W", dtype="float32")
    
    out_size = M + N - 1
    
    k = te.reduce_axis((0, M), name="k")
    
    B = te.compute(
        (out_size,),
        lambda i: te.sum(
            tvm.tir.if_then_else(
                tvm.tir.all(k >= 0, k < M, i-k >= 0, i-k < N),
                A[k] * W[i-k],
                tvm.tir.const(0.0, "float32")
            ),
            axis=k
        ),
        name="B"
    )
    
    s = te.create_schedule(B.op)
    
    i = B.op.axis[0]

    thread_per_block = 256
    max_blocks = 1024
    
    num_blocks = min(max_blocks, (out_size + thread_per_block - 1) // thread_per_block)
    
    block_i, thread_i = s[B].split(i, factor=(out_size + num_blocks - 1) // num_blocks)
    
    s[B].bind(block_i, te.thread_axis("blockIdx.x"))
    s[B].bind(thread_i, te.thread_axis("threadIdx.x"))
    
    return s, A, W, B

# tiling optimization renamed to keep as main 
def make_conv1d_gpu_scheduler(M, N):
    """
    Uses tiling and better memory access patterns
    
    Args:
        M: size of the first input array
        N: size of the second input array
        
    Returns:
        s: schedule
        A: first input placeholder
        W: second input placeholder (weights)
        B: output tensor
    """
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")
    
    k = te.reduce_axis((0, N), name="k")
    
    B = te.compute(
        (M + N - 1,),
        lambda i: te.sum(
            tvm.tir.if_then_else(
                tvm.tir.all(i - k >= 0, i - k < M),
                A[i - k] * W[k],
                0.0
            ),
            axis=k
        ),
        name="B"
    )
    
    s = te.create_schedule(B.op)
    
    i = B.op.axis[0]
    
    BL = s.cache_write(B, "local")
    
    block_size = 256 
    thread_size = 4 
    
    block_i, thread_i = s[B].split(i, factor=block_size)
    thread_i, vec_i = s[B].split(thread_i, factor=thread_size)
    
    s[B].bind(block_i, te.thread_axis("blockIdx.x"))
    s[B].bind(thread_i, te.thread_axis("threadIdx.x"))
    
    s[BL].compute_at(s[B], thread_i)
    
    i_local = BL.op.axis[0]
    k_local = BL.op.reduce_axis[0]
    

    s[BL].unroll(i_local)
    
    if N > 16:
        ko, ki = s[BL].split(k_local, factor=16)
        s[BL].unroll(ki)
    else:
        s[BL].unroll(k_local)
    
    return s, A, W, B


def apply_tiling_optimization(s, C, M, N, K, tile_x=32, tile_y=32, tile_k=32):
    """
    Applies tiling optimization to the GEMM schedule.
    
    This optimization splits the computation into tiles to improve cache locality.
    """
    x, y = s[C].op.axis
    (k,) = s[C].op.reduce_axis
    
    x_outer, x_inner = s[C].split(x, factor=tile_x)
    y_outer, y_inner = s[C].split(y, factor=tile_y)
    k_outer, k_inner = s[C].split(k, factor=tile_k)
    
    s[C].reorder(x_outer, y_outer, k_outer, x_inner, y_inner, k_inner)
    
    return s, (x_outer, y_outer, k_outer, x_inner, y_inner, k_inner)


def gemm_parallel_reduction_optimizer(M, K, N):
    """
    Implements a matrix multiplication schedule with optimized parallel reduction.
    This focuses on optimal reduction order and blocking for GPUs.
    """
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    k = te.reduce_axis((0, K), "k")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    
    s = te.create_schedule(C.op)
    
    x, y = s[C].op.axis
    (k,) = s[C].op.reduce_axis
    
    tile_x = min(16, M)
    tile_y = min(32, N)
    tile_k = min(8, K)
    
    xo, xi = s[C].split(x, factor=tile_x)
    yo, yi = s[C].split(y, factor=tile_y)
    ko, ki = s[C].split(k, factor=tile_k)
    
    s[C].reorder(xo, yo, ko, xi, ki, yi)
    
    xio, xii = s[C].split(xi, factor=4)
    
    s[C].bind(xo, te.thread_axis("blockIdx.x"))
    s[C].bind(yo, te.thread_axis("blockIdx.y"))
    s[C].bind(xio, te.thread_axis("threadIdx.x"))
    s[C].bind(yi, te.thread_axis("threadIdx.y"))
    
    s[C].unroll(xii)
    s[C].unroll(ki)
    
    return s, A, B, C

def gemm_tiling_optimizer(M, K, N):
    """
    Implements a matrix multiplication schedule with only tiling optimization.
    """
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    k = te.reduce_axis((0, K), "k")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    
    s = te.create_schedule(C.op)
    
    bx = te.thread_axis("blockIdx.x")
    by = te.thread_axis("blockIdx.y")
    tx = te.thread_axis("threadIdx.x")
    ty = te.thread_axis("threadIdx.y")
    
    x, y = s[C].op.axis
    xo, xi = s[C].split(x, factor=32)
    yo, yi = s[C].split(y, factor=32)
    
    s[C].bind(xo, bx)
    s[C].bind(yo, by)
    s[C].bind(xi, tx)
    s[C].bind(yi, ty)
    
    return s, A, B, C

def gemm_vectorization_optimizer(M, K, N):
    """
    Implements a matrix multiplication schedule with vectorization.
    """
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    k = te.reduce_axis((0, K), "k")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    
    s = te.create_schedule(C.op)
    
    x, y = s[C].op.axis
    (k,) = s[C].op.reduce_axis
    
    yo, yi = s[C].split(y, factor=4)
    xo, xi = s[C].split(x, factor=32)
    
    s[C].bind(xo, te.thread_axis("blockIdx.x"))
    s[C].bind(yo, te.thread_axis("blockIdx.y"))
    
    s[C].bind(xi, te.thread_axis("threadIdx.x"))
    
    s[C].vectorize(yi)
    
    return s, A, B, C


def make_gemm_gpu_scheduler(M, K, N):
    """
    Matrix multiplication scheduler with configurable optimizations.
    
    Args:
        M, K, N: Matrix dimensions
        use_tiling: Apply tiling optimization
        use_memory: Apply memory/parallel reduction optimization
        use_vectorization: Apply vectorization optimization
    
    Returns:
        The schedule, input tensors A and B, and output tensor C
    """

    use_tiling=True
    use_memory=True
    use_vectorization=True

    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    k = te.reduce_axis((0, K), "k")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    
    s = te.create_schedule(C.op)
    
    x, y = s[C].op.axis
    (k,) = s[C].op.reduce_axis
    
    # If no optimizations
    if not (use_tiling or use_memory or use_vectorization):
        bx = te.thread_axis("blockIdx.x")
        by = te.thread_axis("blockIdx.y")
        s[C].bind(x, bx)
        s[C].bind(y, by)
        return s, A, B, C
    
    # Single optimizations
    if use_tiling and not use_memory and not use_vectorization:
        return gemm_tiling_optimizer(M, K, N)
    
    if use_memory and not use_tiling and not use_vectorization:
        return gemm_parallel_reduction_optimizer(M, K, N)
    
    if use_vectorization and not use_tiling and not use_memory:
        return gemm_vectorization_optimizer(M, K, N)
    
    
    bx = te.thread_axis("blockIdx.x")
    by = te.thread_axis("blockIdx.y")
    tx = te.thread_axis("threadIdx.x")
    ty = te.thread_axis("threadIdx.y")
    
    tile_x = min(16, M)
    tile_y = min(32, N)
    tile_k = min(8, K)
    
    if use_tiling:
        xo, xi = s[C].split(x, factor=tile_x)
        yo, yi = s[C].split(y, factor=tile_y)
        
        if use_memory:
            ko, ki = s[C].split(k, factor=tile_k)
            s[C].reorder(xo, yo, ko, xi, yi, ki)
        else:
            s[C].reorder(xo, yo, xi, yi)
        
        if use_vectorization:
            if tile_y >= 4:
                yi_outer, yi_inner = s[C].split(yi, factor=4)
                s[C].reorder(xo, yo, xi, yi_outer, yi_inner)
                s[C].vectorize(yi_inner)
        
        s[C].bind(xo, bx)
        s[C].bind(yo, by)
        s[C].bind(xi, tx)
        
        if not use_vectorization:
            s[C].bind(yi, ty)
        else:
            if 'yi_outer' in locals():
                s[C].bind(yi_outer, ty)
    else:
        if use_vectorization:
            yo, yi = s[C].split(y, factor=4)
            s[C].vectorize(yi)
        
        s[C].bind(x, bx)
        s[C].bind(y if not use_vectorization else yo, by)
    
    return s, A, B, C

def make_dwsp_conv2d_gpu_scheduler_working(B, C, H, W, K):
    if K % 2 != 1:
        raise ValueError("Kernel size K must be odd")
    
    input_tensor = te.placeholder((B, C, H, W), name="A")
    kernel_tensor = te.placeholder((C, 1, K, K), name="W")
    
    padding = (K - 1) // 2
    
    r_kh = te.reduce_axis((0, K), name="rh")
    r_kw = te.reduce_axis((0, K), name="rw")
    
    output_tensor = te.compute(
        (B, C, H, W),
        lambda n, c, y, x: te.sum(
            te.if_then_else(
                te.all(
                    y - padding + r_kh >= 0,
                    y - padding + r_kh < H,
                    x - padding + r_kw >= 0,
                    x - padding + r_kw < W
                ),
                input_tensor[n, c, y - padding + r_kh, x - padding + r_kw] * kernel_tensor[c, 0, r_kh, r_kw],
                0.0
            ),
            axis=[r_kh, r_kw]
        ),
        name="output"
    )
    
    schedule = te.create_schedule(output_tensor.op)
    
    n_axis, c_axis, y_axis, x_axis = output_tensor.op.axis
    
    y_outer, y_inner = schedule[output_tensor].split(y_axis, nparts=4)
    x_outer, x_inner = schedule[output_tensor].split(x_axis, nparts=16)
    
    schedule[output_tensor].reorder(n_axis, c_axis, y_outer, x_outer, y_inner, x_inner)
    
    schedule[output_tensor].bind(n_axis, te.thread_axis("blockIdx.z"))
    schedule[output_tensor].bind(c_axis, te.thread_axis("blockIdx.y"))
    schedule[output_tensor].bind(y_outer, te.thread_axis("blockIdx.x"))
    schedule[output_tensor].bind(y_inner, te.thread_axis("threadIdx.y"))
    schedule[output_tensor].bind(x_inner, te.thread_axis("threadIdx.x"))
    
    kernel_shared = schedule.cache_read(kernel_tensor, "shared", [output_tensor])
    schedule[kernel_shared].compute_at(schedule[output_tensor], y_outer)
    
    return schedule, input_tensor, kernel_tensor, output_tensor


def conv2d_gpu_optimized_memory(B, C, H, W, K):
    """
    Optimized version 1: Memory Hierarchy Optimization
    
    This implementation focuses on memory access patterns and caching:
    - Uses shared memory to cache kernel weights
    - Uses thread-local registers to cache inputs
    - Optimizes loop ordering for better memory access patterns
    """
    if K % 2 != 1:
        raise ValueError("Kernel size K must be odd")
    
    input_tensor = te.placeholder((B, C, H, W), name="A")
    kernel_tensor = te.placeholder((C, 1, K, K), name="W")
    
    padding = K // 2
    
    r_kh = te.reduce_axis((0, K), name="kh")
    r_kw = te.reduce_axis((0, K), name="kw")
    
    output_tensor = te.compute(
        (B, C, H, W),
        lambda n, c, y, x: te.sum(
            te.if_then_else(
                te.all(
                    y - padding + r_kh >= 0,
                    y - padding + r_kh < H,
                    x - padding + r_kw >= 0,
                    x - padding + r_kw < W
                ),
                input_tensor[n, c, y - padding + r_kh, x - padding + r_kw] * kernel_tensor[c, 0, r_kh, r_kw],
                0.0
            ),
            axis=[r_kh, r_kw]
        ),
        name="output"
    )
    
    schedule = te.create_schedule(output_tensor.op)
    
    n, c, y, x = output_tensor.op.axis
    kh, kw = output_tensor.op.reduce_axis
    
    y_factor = 4
    x_factor = 16
    
    y_outer, y_inner = schedule[output_tensor].split(y, factor=y_factor)
    x_outer, x_inner = schedule[output_tensor].split(x, factor=x_factor)
    
    kh_outer, kh_inner = schedule[output_tensor].split(kh, factor=3)
    kw_outer, kw_inner = schedule[output_tensor].split(kw, factor=3)
    
    schedule[output_tensor].reorder(n, c, y_outer, x_outer, y_inner, kh_outer, kw_outer, x_inner, kh_inner, kw_inner)
    
    schedule[output_tensor].bind(n, te.thread_axis("blockIdx.z"))
    schedule[output_tensor].bind(c, te.thread_axis("blockIdx.y"))
    schedule[output_tensor].bind(y_outer, te.thread_axis("blockIdx.x"))
    schedule[output_tensor].bind(y_inner, te.thread_axis("threadIdx.y"))
    schedule[output_tensor].bind(x_inner, te.thread_axis("threadIdx.x"))
    
    kernel_shared = schedule.cache_read(kernel_tensor, "shared", [output_tensor])
    schedule[kernel_shared].compute_at(schedule[output_tensor], y_outer)
    
    input_local = schedule.cache_read(input_tensor, "local", [output_tensor])
    schedule[input_local].compute_at(schedule[output_tensor], kh_outer)
    
    return schedule, input_tensor, kernel_tensor, output_tensor

def conv2d_gpu_scheduler_coop_fetching(B, C, H, W, K):
    assert K % 2 == 1 
    pad = K // 2 
    
    inp = te.placeholder((B, C, H, W), name="A")
    ker = te.placeholder((C, 1, K, K), name="W")
    
    kh = te.reduce_axis((-pad, pad + 1), "kh")
    kw = te.reduce_axis((-pad, pad + 1), "kw")
    
    def valid_index(b, c, h, w):
        return te.if_then_else(
            te.all(h >= 0, h < H, w >= 0, w < W),
            inp[b, c, h, w],
            0.0
        )
    
    out = te.compute(
        (B, C, H, W),
        lambda b, c, h, w: te.sum(
            valid_index(b, c, h + kh, w + kw) * ker[c, 0, kh + pad, kw + pad],
            axis=[kh, kw],
        ),
        name="out",
    )
    
    s = te.create_schedule(out.op)
    
    out_local = s.cache_write(out, "local")
    ker_shared = s.cache_read(ker, "shared", [out_local])
    
    b, c, h, w = out.op.axis
    
    bx, tx = s[out].split(h, factor=4)
    by, ty = s[out].split(w, factor=8)
    
    s[out].reorder(b, c, bx, by, tx, ty)
    
    s[out].bind(bx, te.thread_axis("blockIdx.x"))
    s[out].bind(tx, te.thread_axis("threadIdx.x"))
    s[out].bind(c, te.thread_axis("blockIdx.y"))
    s[out].bind(ty, te.thread_axis("threadIdx.y"))
    s[out].bind(b, te.thread_axis("blockIdx.z"))
    
    s[out_local].compute_at(s[out], ty)
    
    s[ker_shared].compute_at(s[out], bx)
    
    return s, inp, ker, out

def conv2d_gpu_scheduler_tile_based(B, C, H, W, K):
    """
    Args:
        B: Batch size
        C: Number of channels
        H: Height of input
        W: Width of input
        K: Kernel size (must be odd)

    Returns:
        schedule, input_tensor, kernel_tensor, output_tensor
    """
    assert K % 2 == 1
    pad = K // 2
    
    inp = te.placeholder((B, C, H, W), name="A")
    ker = te.placeholder((C, 1, K, K), name="W")
    
    kh = te.reduce_axis((-pad, pad + 1), "kh")
    kw = te.reduce_axis((-pad, pad + 1), "kw")
    
    def valid_index(b, c, h, w):
        return te.if_then_else(
            te.all(h >= 0, h < H, w >= 0, w < W),
            inp[b, c, h, w],
            0.0
        )
    
    out = te.compute(
        (B, C, H, W),
        lambda b, c, h, w: te.sum(
            valid_index(b, c, h + kh, w + kw) * ker[c, 0, kh + pad, kw + pad],
            axis=[kh, kw],
        ),
        name="out",
    )
    
    s = te.create_schedule(out.op)
    
    out_local = s.cache_write(out, "local")
    inp_shared = s.cache_read(inp, "shared", [out_local])
    ker_shared = s.cache_read(ker, "shared", [out_local])
    
    b, c, h, w = out.op.axis

    h_factor = min(8, H) 
    w_factor = min(8, W)
    
    bx, tx = s[out].split(h, factor=h_factor)
    by, ty = s[out].split(w, factor=w_factor)
    
    s[out].reorder(b, c, bx, by, tx, ty)
    
    s[out].bind(bx, te.thread_axis("blockIdx.x"))
    s[out].bind(tx, te.thread_axis("threadIdx.x"))
    s[out].bind(c, te.thread_axis("blockIdx.y"))
    s[out].bind(ty, te.thread_axis("threadIdx.y"))
    s[out].bind(b, te.thread_axis("blockIdx.z"))
    
    s[out_local].compute_at(s[out], ty)
    s[inp_shared].compute_at(s[out], bx)
    s[ker_shared].compute_at(s[out], bx)
    
    return s, inp, ker, out

def make_dwsp_conv2d_gpu_scheduler(B, C, H, W, K):
    """
    Args:
        B: Batch size
        C: Number of channels
        H: Height of input
        W: Width of input
        K: Kernel size (must be odd)
    
    Returns:
        schedule, input_tensor, kernel_tensor, output_tensor
    """
    assert K % 2 == 1, "Kernel size must be odd"
    pad = K // 2
    
    inp = te.placeholder((B, C, H, W), name="A")
    ker = te.placeholder((C, 1, K, K), name="W")
    
    kh = te.reduce_axis((-pad, pad + 1), "kh")
    kw = te.reduce_axis((-pad, pad + 1), "kw")
    
    def valid_index(b, c, h, w):
        return te.if_then_else(
            te.all(h >= 0, h < H, w >= 0, w < W),
            inp[b, c, h, w],
            0.0
        )
    
    out = te.compute(
        (B, C, H, W),
        lambda b, c, h, w: te.sum(
            valid_index(b, c, h + kh, w + kw) * ker[c, 0, kh + pad, kw + pad],
            axis=[kh, kw],
        ),
        name="out",
    )
    
    s = te.create_schedule(out.op)
    
    out_local = s.cache_write(out, "local")
    ker_shared = s.cache_read(ker, "shared", [out_local])
    inp_shared = s.cache_read(inp, "shared", [out_local]) 
    
    b, c, h, w = out.op.axis

    h_factor = min(4, H)
    w_factor = min(8, W)
    
    bx, tx = s[out].split(h, factor=h_factor)
    by, ty = s[out].split(w, factor=w_factor)
    
    s[out].reorder(b, c, bx, by, tx, ty)
    
    s[out].bind(bx, te.thread_axis("blockIdx.x"))
    s[out].bind(tx, te.thread_axis("threadIdx.x"))
    s[out].bind(c, te.thread_axis("blockIdx.y"))
    s[out].bind(ty, te.thread_axis("threadIdx.y"))
    s[out].bind(b, te.thread_axis("blockIdx.z"))

    s[out_local].compute_at(s[out], ty)
    s[inp_shared].compute_at(s[out], bx)
    s[ker_shared].compute_at(s[out], bx)

    if H >= 16 and W >= 16 and K >= 3:
        inp_local = s.cache_read(inp_shared, "local", [out_local])
        s[inp_local].compute_at(s[out_local], s[out_local].op.axis[0])

    if K >= 5:
        kh_axis, kw_axis = out_local.op.reduce_axis

        local_axes = s[out_local].op.axis
        
        if len(local_axes) >= 4:
            s[out_local].reorder(
                local_axes[0], local_axes[1], local_axes[2], local_axes[3],
                kh_axis, kw_axis
            )
    
    return s, inp, ker, out