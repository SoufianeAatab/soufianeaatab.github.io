---
layout: post
title: "Autodiff: Generating C Code for Neural Network Training"
date: 2026-02-06
categories: [machine-learning, embedded-systems, c]
tags: [autodiff, neural-networks, iot, code-generation]
---

## Introduction

During my thesis on "Training Neural Networks on IoT Devices," I spent significant time manually implementing forward and backward passes for neural network operators in C. Each operator required careful implementation of both the computation and its gradient, which was time-consuming and error-prone. This experience motivated me to create a tool that automates this process.

Autodiff is an automatic differentiation framework that generates standalone C code for training neural networks. Unlike traditional autodiff libraries that compute gradients at runtime, this framework computes the backward pass during code generation, eliminating runtime overhead.

## Motivation

Training neural networks on resource-constrained devices presents unique challenges. However, the primary problem this framework addresses is not optimization, but automation.

**The core problem**: Manually implementing gradient propagation through neural network layers is tedious and error-prone. For each operator, you must:
- Implement the forward pass
- Derive the backward pass mathematically
- Implement gradient computation correctly
- Wire gradients between layers without mistakes

This is repetitive work that can be automated.

**What this framework does**: Automatically generates the backward pass and handles gradient propagation between layers. The generated C operators are basic implementations - not optimized for memory or speed.

**What this framework does not do**: Produce highly optimized C code. If you need performance-critical implementations, you can replace the basic operators with your own optimized versions. The framework handles the gradient wiring; you handle the performance.

## How It Works

The framework operates in two phases:

### 1. Graph Construction (Python)

Define your model using a simple Python API:

```python
# Define model parameters
w1 = Param(None, shape=(32, 28*28), var_name='w1')
w2 = Param(None, shape=(10, 32), var_name='w2')

# Forward pass
z = matmul(input, w1.t())
a = sigmoid(z)
z2 = matmul(a, w2.t())
a2 = log_softmax(z2)
loss = nll_loss(a2, output)
```

### 2. Code Generation (Compile-time)

The framework analyzes the computation graph and generates optimized C code:

```c
// Forward pass
mat_mul(&input_ptr[l * 784], &buf[0], &buf[26202], ...);
sigmoid(&buf[26202], &buf[26234], 32);
mat_mul(&buf[26234], &buf[25088], &buf[26266], ...);
log_softmax(&buf[26266], &buf[26276], 10);

// Backward pass (automatically generated)
// ... gradient computations ...

// Parameter updates
for (uint32_t k=0; k<25088; ++k) {
    buf[0 + k] -= buf[26420 + k] * lr;
}
```

The generated code is self-contained and can be compiled with any C compiler.

## Key Features

**Automated Gradient Propagation**: The framework automatically computes gradients and wires them between layers. You don't need to manually implement backward passes or track gradient flow.

**Compile-time Autodiff**: The backward graph is computed once during code generation. No runtime autodiff overhead.

**Bring Your Own Operators**: The generated code uses basic C implementations of operators (matmul, sigmoid, etc.). These are not optimized. If you're experienced with optimizing neural network operators in C (e.g., using SIMD, loop unrolling, specialized libraries), you can replace them with your own implementations. The framework handles the gradient connections - you handle the performance.

**Minimal Dependencies**: Generated code requires only standard C libraries. No external autodiff runtime needed.

## Implementation Details

### Computation Graph

The framework builds a directed acyclic graph (DAG) representing the computation:

```python
graph = backward(loss, [w1.id, w2.id])
ops = graph.build()
```

Each node represents an operation (matmul, sigmoid, etc.) and stores:
- Input dependencies
- Output shape
- Buffer allocation information
- Backward pass implementation

### Memory Management

All intermediate values are stored in a single pre-allocated buffer:

```c
float* buf = (float*)calloc(51828, sizeof(float));
```

Buffer offsets are assigned during code generation. The allocation strategy is basic and not optimized for minimal memory usage.

### Gradient Computation

For each operation in the forward pass, the framework generates the corresponding backward operation. For example, sigmoid:

```c
// Forward: y = sigmoid(x)
sigmoid(&buf[input], &buf[output], size);

// Backward: dx = dy * sigmoid'(x)
sigmoid_diff(&buf[input], &buf[grad_output], &buf[grad_input], size);
```

### Operator Customization

The generated code calls basic operator implementations. For example, `mat_mul` is a simple triple-loop implementation with no optimizations:

```c
mat_mul(a, b, c, rows, cols, ...);  // Basic implementation
```

If you need optimized operators, replace them with your own:
- Hand-tuned SIMD implementations
- Hardware-specific optimizations (ARM NEON, CMSIS-NN)
- Vendor libraries (BLAS, cuBLAS)

The framework doesn't care about the operator implementation - it only needs the function signature to match. This separation allows domain experts in low-level optimization to focus on performance-critical kernels while the framework handles the tedious gradient bookkeeping.

## Example: MNIST Training

See the [complete example](https://github.com/SoufianeAatab/autodiff/blob/master/examples/fc_mnist.py) in the repository.

The framework generates training code for a simple fully-connected network:
- Input: 28×28 images (784 features)
- Hidden layer: 32 units with sigmoid activation
- Output: 10 classes with log-softmax
- Loss: Negative log-likelihood

The generated C code includes forward pass, backward pass, and SGD optimizer, ready to compile and run on embedded devices.

```c
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "ops.cc"

int main() {
//===================================================
float lr = 0.01;
float* buf = (float*)calloc(51828, sizeof(float));

float temp_1[25088] = {0.08822693, ... , 0.084507786, }; // Weights layer_1 are automatically generated from python in this case (784*32)
memcpy(&buf[794], temp_1, sizeof(float) * 25088 );

float temp_2[320] = {0.07258509, ... , 0.071656786, }; // Weights layer_2 are automatically generated from python in this case (32*10)
memcpy(&buf[25882], temp_2, sizeof(float) * 320 );

size_t epochs = 5;
for (size_t i = 0; i < epochs; ++i) {
    float loss = 0;
    for (size_t j = 0; j < 60000; ++j) {
    
        //buf[0] = Note: this ptr is where the input data should be.
        //buf[784] = output; // Note: this ptr is where output data should be.
        
        mat_mul(&buf[0] /* (1, 784) */, &buf[794] /* (784, 32) */, &buf[26202] /* (1, 32) */, 1, 784, 784, 1, 784, 32, 1, 784); // (1, 32) 5
        sigmoid(&buf[26202] /* (1, 32)*/ , &buf[26234] /*(1, 32)*/, 32); // (1, 32) 6
        mat_mul(&buf[26234] /* (1, 32) */, &buf[25882] /* (32, 10) */, &buf[26266] /* (1, 10) */, 1, 32, 32, 1, 32, 10, 1, 32); // (1, 10) 8
        log_softmax(&buf[26266], &buf[26276], 10); // (1, 10) 9
        exp_(&buf[26276], &buf[26286], 10); // (1, 10) 15
        nll_loss(&buf[26276], &buf[784], &buf[26296], 10);
        loss+=buf[26296]; // (1, 10) 10
        for(uint32_t k=0;k<10;++k){
        	buf[26306 + k] = 1.0f;}
        for(uint32_t k=0;k<10;++k){
        	buf[26316 + k] = -1;
        }
        mul(&buf[26306], &buf[26316], &buf[26326], 10); // (1, 10) 13
        mul(&buf[26326], &buf[784], &buf[26336], 10); // (1, 10) 14
        add(&buf[26286], &buf[26336], &buf[26346], 10); // (1, 10) 16
        mat_mul(&buf[26346] /* (10, 1) */, &buf[26234] /* (1, 32) */, &buf[26356] /* (10, 32) */, 10, 1, 1, 10, 1, 32, 32, 1); // (10, 32) 19
        mat_mul(&buf[26346] /* (1, 10) */, &buf[25882] /* (10, 32) */, &buf[26676] /* (1, 32) */, 1, 10, 10, 1, 10, 32, 32, 1); // (1, 32) 20
        sigmoid_diff(&buf[26202], &buf[26676], &buf[26708], 32); // (1, 32) 21
        mat_mul(&buf[26708] /* (32, 1) */, &buf[0] /* (1, 784) */, &buf[26740] /* (32, 784) */, 32, 1, 1, 32, 1, 784, 784, 1); // (32, 784) 24
        // sgd for 24
        for (uint32_t k=0;k<25088;++k){
        	buf[794 + k] -= buf[26740 + k] * lr;
        }
        // sgd for 19
        for (uint32_t k=0;k<320;++k){
        	buf[25882 + k] -= buf[26356 + k] * lr;
        }
        }
        printf("Loss: %f\n", loss / 60000.0f);
    }
    return 0;
}
```

## Current Status

**Implemented operators:**
- MatMul [DONE]
- Sigmoid, ReLU, Tanh [DONE]
- LogSoftmax [DONE]
- NLL Loss [DONE]
- Conv2d, MaxPool2d, MSE Loss [WIP]

**Testing**: Gradients are validated against PyTorch to ensure correctness.

## Validation

The framework validates generated gradients by comparing them against PyTorch's autograd:

```python
# Generate gradients with autodiff
# Compare with PyTorch ground truth
# Verify differences are within floating-point tolerance
```

This ensures the backward pass computes correct gradients for all operators.

## Technical Challenges

**Gradient flow tracking**: Determining which gradients flow to which buffers, especially when operations share inputs or have multiple outputs.

**Buffer allocation**: Assigning buffer offsets without conflicts while minimizing total memory usage.

**Shape inference**: Propagating tensor shapes through the graph to generate correct array indexing.

**Backward pass derivation**: Implementing correct gradient formulas for each operator (chain rule application).

**Code generation**: Producing readable, properly indented C code from Python.

Note that these challenges are about correctness and automation, not performance optimization.

## Future Work

- Complete Conv2d and MaxPool2d implementations
- Add more optimizers (Adam, RMSprop)
- Optimize buffer allocation algorithm
- Support for dynamic batch sizes

## Conclusion

Autodiff automates the tedious work of implementing and wiring backward passes for neural networks in C. The goal is not to generate the fastest or most memory-efficient code, but to eliminate manual gradient computation and error-prone bookkeeping.

If you need optimized operators, you can implement them separately. The framework handles what's automatable (gradient propagation) so you can focus on what requires expertise (performance optimization).

The project is open source and available on [GitHub](https://github.com/SoufianeAatab/autodiff). It remains a side project for exploring ideas in code generation and automatic differentiation.

## References

- [Repository](https://github.com/SoufianeAatab/autodiff)
- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
