#include <iostream>
#include <fstream>
#include"ray.h"
#include"vec.h"
#include"hittable.h"
#include"hittable_list.h"
#include"sphere.h"
#include "camera.h"
#include "material.h"

class camera;

__device__ vec ray_color(const ray& r , hittable** world , curandState* local_state){
    vec curr_attenuated(1.0f ,1.0f,1.0f);
    ray curr_ray = r;

    for(int i = 0 ; i < 30 ;i++){
        hit_record rec;

        if((*world)->hit(curr_ray , 0.001f , 100000.0f , rec)){
            ray scattered ;
            vec attenuated;

            if(rec.mat->scatter(curr_ray,rec , attenuated ,scattered , local_state)){
                curr_attenuated *= attenuated;  // multiply the curr color with the obj's color
                
                curr_ray = scattered;   // here the scattered ray becomes the current ray
            }else{
                return vec(0.0f, 0.0f, 0.0f);
            }
        }else{
            // If it didn't hit the sphere, draw the sky gradient
            vec unit_direction = unit_vector(curr_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            return  (1.0f - t) * curr_attenuated * vec(1.0f , 1.0f , 1.0f) + t * vec(0.5f , 0.7f , 1.0f) ; 
        }
    }
    // return black as even after many bounces it didnt reach the sky --> it is trapped in some dark place
    return vec(0.0f, 0.0f, 0.0f);
}
__global__ void render_kernel(vec* fb , int max_x , int max_y , vec lower_left_corner , vec horizontal ,vec vertical , vec origin, camera** cam , hittable** world , curandState* local_state){
    int i = blockDim.x * blockIdx.x + threadIdx.x ;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if((i >= max_x) || (j >= max_y)) return;

    float u = float(i) / float(max_x - 1);
    float v = float(j) / float(max_y - 1);
    
    ray r = (*cam)->get_ray(u,v);

    int pixel_idx = j * max_x + i;
    fb[pixel_idx] = ray_color(r , world , local_state);;    
}
__global__ void create_world(hittable** list , hittable** world){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        list[0] = new sphere(vec(0.0f , -100.5f, -1.0f) , 100.0f);
        list[1] = new sphere(vec(0.0f ,0.0f , -1.0f), 0.5f);

        *world = new  hittable_list(list , 2);
    }
}
__global__ void clear_world(hittable** list , hittable** world){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        delete((sphere*)list[0]);
        delete((sphere*)list[1]);
        delete(*world);
    }
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
    vec lower_left_corner = (origin - (horizontal / 2.0f) - (vertical / 2.0f) - vec(0.0f , 0.0f , focal_length));
    
    
    vec* h_fb = (vec*)malloc(fb_size);
    vec* d_fb;
    cudaMalloc((void**)&d_fb , fb_size);

    int tx = 8;
    int ty = 8;
    dim3 threads(tx , ty);
    dim3 blocks((image_width / tx) +1 , (image_height / ty)+1);

    hittable** list;
    cudaMalloc((void**)&list , 2* sizeof(hittable*));
    hittable** world;
    cudaMalloc((void**)&world , sizeof(hittable*));

    create_world <<< 1 , 1>>>(list ,world) ;
    cudaDeviceSynchronize();

    render_kernel<<< blocks , threads >>>(d_fb , image_width , image_height , lower_left_corner , horizontal , vertical , origin , world);
    cudaDeviceSynchronize();

    cudaMemcpy(h_fb , d_fb , fb_size , cudaMemcpyDeviceToHost);

    std::ofstream file("image.ppm");
    file << "P3\n" << image_width << " " << image_height << "\n255\n";
    
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

    clear_world <<< 1,1 >>>(list , world);
    cudaDeviceSynchronize();

    cudaFree(d_fb);
    free(h_fb);
    cudaFree(world);

    std::cout << "Successfully rendered image.ppm!" << std::endl;

    return 0;

}