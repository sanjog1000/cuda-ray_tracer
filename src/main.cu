#include <iostream>
#include <fstream>
#include<vector>
#include"ray.h"
#include"vec.h"
#include "random.h"
#include"hittable.h"
#include"hittable_list.h"
#include"sphere.h"
#include "material.h"
#include "camera.h"

__device__ vec ray_color(const ray& r , hittable** world , curandState* local_state){
    vec curr_attenuated(1.0f ,1.0f,1.0f);
    ray curr_ray = r;

    for(int i = 0 ; i < 30 ;i++){
        hit_record rec;

        if((*world)->hit(curr_ray , 0.001f , 100000.0f , rec , local_state)){
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
            return  (1.0f - t) * curr_attenuated * vec(1.0f , 1.0f , 1.0f) + t * curr_attenuated * vec(0.5f , 0.7f , 1.0f) ; 
        }
    }
    // return black as even after many bounces it didnt reach the sky --> it is trapped in some dark place
    return vec(0.0f, 0.0f, 0.0f);
}
__global__ void init_random_states(curandState* states , int max_x , int max_y){
    int i = blockDim.x * blockIdx.x + threadIdx.x ;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if((i >= max_x) || (j >= max_y)) return;
    int pixel_idx = j * max_x + i;

    curand_init(
        1234ULL, // seed
        pixel_idx, // unique sequence per pixel
        0,   // offset
        &states[pixel_idx]
    );
}
__global__ void render_kernel(vec* fb , int max_x , int max_y , camera cam , hittable** world , curandState* states){
    int i = blockDim.x * blockIdx.x + threadIdx.x ;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if((i >= max_x) || (j >= max_y)) return;

    int pixel_idx = j * max_x + i;

    // copy this pixel's rng state into a local register/state variable
    curandState local_state = states[pixel_idx];

    float u = float(i) / float(max_x - 1);
    float v = float(j) / float(max_y - 1);
    
    ray r = cam.get_ray(u,v,&local_state);

    fb[pixel_idx] = ray_color(r , world , &local_state);

    // save the updated rng state back
    states[pixel_idx] = local_state;
}

enum MaterialType {MAT_LAMBERTIAN, MAT_METAL, MAT_DIELECTRIC};
struct sphereDesc{
    vec centre;
    float radius ;
    MaterialType mat_type;
    vec albedo;     // used by lambertian , metal
    float param;    // fuzz for metal . ir for dielectric
};

__global__ void create_world(hittable** list , hittable** world , material** materials ,sphereDesc* desc , int count){
    if(threadIdx.x != 0 || blockIdx.x != 0) return;

    for(int i = 0  ; i < count ; i++){
        material*m = nullptr;
        switch(desc[i].mat_type){
            case MAT_LAMBERTIAN :
                m = new lambertian(desc[i].albedo);
                break;
            
            case MAT_METAL:
                m = new metal(desc[i].albedo, desc[i].param);
                break;

            case MAT_DIELECTRIC:
                m = new dielectric(desc[i].param);
                break;  
        }
        materials[i] = m;
        list[i] = new sphere(desc[i].centre , desc[i].radius , m);
    }
    *world = new hittable_list(list , count);
}
__global__ void clear_world(hittable** list, hittable** world, material** materials, int count) {
    if(threadIdx.x != 0 || blockIdx.x != 0) return;

    for(int i = 0; i < count; i++) {
        delete list[i];
        delete materials[i];
    }
    delete *world;
}

int main(){
    const float aspect_ratio = 16.0f / 9.0f ;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio) ;

    const int total_pixels = image_height* image_width;
    const size_t fb_size = total_pixels * sizeof(vec);
     
    cudaDeviceSetLimit(cudaLimitMallocHeapSize , 256*1024*1024);

    vec lookfrom(0.0f, 0.0f, 0.0f);
    vec lookat(0.0f, 0.0f, -1.0f);
    vec vup(0.0f, 1.0f, 0.0f);

    const float vfov = 90.0f;
    const float aperture = 0.0f;
    const float focus_dist = 1.0f;

    camera cam(lookfrom , lookat , vup , vfov , aspect_ratio , aperture , focus_dist);

    std::vector<sphereDesc> h_scene = {
        { vec(0.0f, -100.5f, -1.0f), 100.0f, MAT_LAMBERTIAN, vec(0.8f, 0.8f, 0.0f), 0.0f },
        { vec(0.0f,    0.0f, -1.0f),   0.5f, MAT_LAMBERTIAN, vec(0.7f, 0.3f, 0.3f), 0.0f },
    };

    const int object_count = static_cast<int>(h_scene.size());

    sphereDesc* d_descs;
    cudaMalloc((void**)&d_descs , object_count * sizeof(sphereDesc));
    cudaMemcpy(d_descs, h_scene.data(), object_count * sizeof(sphereDesc), cudaMemcpyHostToDevice);
    //    Memory Allocation
    vec* h_fb = (vec*)malloc(fb_size);
    vec* d_fb;
    cudaMalloc((void**)&d_fb , fb_size);

    curandState* d_states;
    cudaMalloc((void**)&d_states , total_pixels * sizeof(curandState));

    hittable** d_list;
    cudaMalloc((void**)&d_list , object_count * sizeof(hittable*));
    
    material** d_materials;
    cudaMalloc((void**)&d_materials , object_count * sizeof(material*));
    
    hittable** d_world;
    cudaMalloc((void**)&d_world , sizeof(hittable*));

    // Kernel Setup
    int tx = 8;
    int ty = 8;
    dim3 threads(tx , ty);
    dim3 blocks((image_width + tx - 1)/tx , (image_height + ty -1) / ty);
 
    create_world <<< 1 , 1>>>(d_list , d_world , d_materials , d_descs , object_count) ;
    cudaDeviceSynchronize();

    init_random_states<<< blocks, threads>>>(d_states , image_width , image_height);
    cudaDeviceSynchronize();

    render_kernel<<< blocks , threads >>>(d_fb , image_width , image_height , cam , d_world , d_states);
    cudaDeviceSynchronize();

    cudaMemcpy(h_fb , d_fb , fb_size , cudaMemcpyDeviceToHost);

    //  Write PPM
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

    clear_world <<< 1,1 >>>(d_list , d_world , d_materials, object_count);
    cudaDeviceSynchronize();

    cudaFree(d_fb);
    cudaFree(d_world);
    cudaFree(d_list);
    cudaFree(d_states);
    cudaFree(d_materials);
    cudaFree(d_descs);
    free(h_fb);

    std::cout << "Successfully rendered image.ppm!" << std::endl;

    return 0;
}