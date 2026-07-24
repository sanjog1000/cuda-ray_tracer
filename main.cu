#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

// CUDA Kernel: Executes on the GPU in parallel across all pixels
__global__ void render_kernel(unsigned char* fb, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel_index = (y * width + x) * 3;

    // Normalize pixel coordinates to [0, 1] for a color gradient
    float r = (float)x / (float)width;
    float g = (float)y / (float)height;
    float b = 0.2f;

    fb[pixel_index + 0] = static_cast<unsigned char>(255.0f * r); // Red
    fb[pixel_index + 1] = static_cast<unsigned char>(255.0f * g); // Green
    fb[pixel_index + 2] = static_cast<unsigned char>(255.0f * b); // Blue
}

int main() {
    int width = 800;
    int height = 600;
    size_t fb_size = width * height * 3 * sizeof(unsigned char);

    // 1. Allocate CPU (Host) and GPU (Device) memory
    unsigned char *h_fb = (unsigned char*)malloc(fb_size);
    unsigned char *d_fb;
    cudaMalloc((void**)&d_fb, fb_size);

    // 2. Define GPU execution configuration (16x16 thread blocks)
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, 
                (height + threads.y - 1) / threads.y);

    // 3. Launch kernel on the GPU
    render_kernel<<<blocks, threads>>>(d_fb, width, height);
    cudaDeviceSynchronize();

    // 4. Copy rendered framebuffer back to CPU memory
    cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost);

    // 5. Output result to a PPM file
    std::ofstream file("output.ppm");
    file << "P3\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height * 3; i += 3) {
        file << (int)h_fb[i] << " " << (int)h_fb[i+1] << " " << (int)h_fb[i+2] << "\n";
    }
    file.close();

    std::cout << "Render complete! Saved to output.ppm\n";

    // Clean up memory
    cudaFree(d_fb);
    free(h_fb);

    return 0;
}