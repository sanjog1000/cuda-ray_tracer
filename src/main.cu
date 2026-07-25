#include <iostream>
#include <fstream>
#include"ray.h"
#include"vec.h"

__device__ vec ray_color(const ray& r){
    vec unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return  vec(1.0f , 1.0f , 1.0f) * (1.0f - t)  +  vec(0.5f , 0.7f , 1.0f) * t; 
}
__global__ void render_kernel(vec* fb , int max_x , int max_y , vec lower_left_corner , vec horizontal ,vec vertical , vec origin){
    int i = blockDim.x * blockIdx.x + threadIdx.x ;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if((i >= max_x) || (j >= max_y)) return;

    float u = float(i) / float(max_x - 1);
    float v = float(j) / float(max_y - 1);

    ray r(origin , lower_left_corner + horizontal*u + vertical*v - origin);

    int pixel_idx = j * max_x + i;
    fb[pixel_idx] = ray_color(r);    
}
int main(){
    const float aspect_ratio = 16.0f / 9.0f ;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio) ;

    int total_pixels = image_height* image_width;
    int fb_size = total_pixels * sizeof(vec);

    float viewport_height = 2.0f;
    float viewport_width = aspect_ratio * viewport_height;
    float focal_length = 1.0f;


    vec origin(0.0f , 0.0f , 0.0f);
    vec horizontal(viewport_width , 0.0f , 0.0f);
    vec vertical(0.0f , viewport_height , 0.0f);
    vec lower_left_corner = (origin - (horizontal / 2.0f) , (vertical / 2.0f) - vec(0.0f , 0.0f , focal_length));
    
    
    vec* h_fb = (vec*)malloc(fb_size);
    vec* d_fb;
    cudaMalloc((void**)&d_fb , fb_size);

    int tx = 8;
    int ty = 8;
    dim3 threads(tx , ty);
    dim3 blocks((image_width / tx) +1 , (image_height / ty)+1);

    render_kernel<<< blocks , threads >>>(d_fb , image_width , image_height , lower_left_corner , horizontal , vertical , origin);
    cudaDeviceSynchronize();

    cudaMemcpy(h_fb , d_fb , fb_size , cudaMemcpyDeviceToHost);

    for (int j = image_height - 1; j >= 0; --j) {   //start from top-left
        for (int i = 0; i < image_width; ++i) {
            int pixel_index = j * image_width + i;
            vec pixel_color = h_fb[pixel_index];
            
            int ir = static_cast<int>(255.999f * pixel_color.x());
            int ig = static_cast<int>(255.999f * pixel_color.y());
            int ib = static_cast<int>(255.999f * pixel_color.z());

            file << ir << " " << ig << " " << ib << "\n";
        }
    }

    file.close();

    cudaFree(d_fb);
    free(h_fb);

    std::cout << "Successfully rendered image.ppm!" << std::endl;

    return 0;

}