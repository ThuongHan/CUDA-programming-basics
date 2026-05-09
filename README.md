# CUDA Experiments on NVIDIA GPUs

This repository contains my experiments with NVIDIA GPU programming using **CUDA C++**.

The goal of this project is to explore GPU computing concepts, performance optimization, and parallel programming techniques by implementing and testing different CUDA-based kernels and applications.

## What this repo includes

- CUDA C++ implementations of basic and advanced GPU kernels  
- Experiments with memory management (host/device memory, unified memory)  
- Performance comparisons between CPU and GPU implementations  
- Small test projects to understand parallel computation patterns  

## Requirements

To run the code in this repository, you need:

- NVIDIA GPU with CUDA support  
- CUDA Toolkit installed  
- A compatible C++ compiler (e.g., `nvcc`)

## Running on Snellius HPC Cluster

In addition to local CUDA setup, this project is also executed on the **Snellius high-performance computing (HPC) cluster**, where NVIDIA GPUs are available through job scheduling.

I connect to the cluster using SSH and submit GPU jobs using batch scripts (`.sh` files) that request GPU resources from the scheduler.

These scripts handle:
- Requesting GPU nodes
- Allocating compute resources via Slurm
- Running CUDA executables on the cluster
- Managing job submission and output logs

This setup allows experiments to be run on real GPU hardware at scale rather than only on local machines.

## Purpose

This repository is mainly for learning and experimentation. It is not intended to be production-ready code, but rather a space to understand how GPU acceleration works in practice.

## Notes

Performance results and implementations may change as I continue experimenting and learning more about CUDA programming.
