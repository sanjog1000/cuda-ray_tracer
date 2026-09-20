#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include "ray.h"
#include "vec.h"
#include "random.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "bvh_node.h"
#include "material.h"
#include "pdf.h"
#include "camera.h"
#include "cuboid.h"
#include "constant_medium.h"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " - " << cudaGetErrorString(err__) << std::endl;       \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

inline float random_range(float lo, float hi){
    return lo + (hi - lo) * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
}

inline float de_nan(float f){
    return (f == f) ? f : 0.0f;
}
inline float clamp01(float f){
    return f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
}

class rotate_y : public hittable {
public:
    hittable* obj;
    vec pivot;
    float sin_theta, cos_theta;

    __host__ __device__ rotate_y(hittable* o, vec pivot_, float degrees) : obj(o), pivot(pivot_) {
        float radians = degrees * 3.14159265359f / 180.0f;
        sin_theta = sinf(radians);
        cos_theta = cosf(radians);
    }

    __host__ __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* local_state) const override {
        vec d = r.origin() - pivot;
        vec local_o(
            cos_theta*d.x() - sin_theta*d.z(),
            d.y(),
            sin_theta*d.x() + cos_theta*d.z()
        );
        vec dir = r.direction();
        vec local_d(
            cos_theta*dir.x() - sin_theta*dir.z(),
            dir.y(),
            sin_theta*dir.x() + cos_theta*dir.z()
        );

        ray local_ray(local_o + pivot, local_d);
        if(!obj->hit(local_ray, t_min, t_max, rec, local_state)) return false;

        vec p = rec.p - pivot;
        vec n = rec.normal;
        vec world_p(
            cos_theta*p.x() + sin_theta*p.z(),
            p.y(),
            -sin_theta*p.x() + cos_theta*p.z()
        );
        vec world_n(
            cos_theta*n.x() + sin_theta*n.z(),
            n.y(),
            -sin_theta*n.x() + cos_theta*n.z()
        );

        rec.p = world_p + pivot;
        rec.normal = world_n;
        return true;
    }

    __host__ __device__ virtual bool bounding_box(aabb& output_box) const override {
        aabb local_box;
        if(!obj->bounding_box(local_box)) return false;

        vec mn( 1e30f,  1e30f,  1e30f);
        vec mx(-1e30f, -1e30f, -1e30f);

        for(int i = 0; i < 2; i++){
            for(int j = 0; j < 2; j++){
                for(int k = 0; k < 2; k++){
                    float x = i ? local_box.max().x() : local_box.min().x();
                    float y = j ? local_box.max().y() : local_box.min().y();
                    float z = k ? local_box.max().z() : local_box.min().z();

                    vec p = vec(x,y,z) - pivot;
                    vec world_corner = pivot + vec(
                        cos_theta*p.x() + sin_theta*p.z(),
                        p.y(),
                        -sin_theta*p.x() + cos_theta*p.z()
                    );

                    mn = vec(fminf(mn.x(), world_corner.x()), fminf(mn.y(), world_corner.y()), fminf(mn.z(), world_corner.z()));
                    mx = vec(fmaxf(mx.x(), world_corner.x()), fmaxf(mx.y(), world_corner.y()), fmaxf(mx.z(), world_corner.z()));
                }
            }
        }
        output_box = aabb(mn, mx);
        return true;
    }
};

class rotate_z : public hittable {
public:
    hittable* obj;
    vec pivot;
    float sin_theta, cos_theta;

    __host__ __device__ rotate_z(hittable* o, vec pivot_, float degrees) : obj(o), pivot(pivot_) {
        float radians = degrees * 3.14159265359f / 180.0f;
        sin_theta = sinf(radians);
        cos_theta = cosf(radians);
    }

    __host__ __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* local_state) const override {
        vec d = r.origin() - pivot;
        vec local_o(
            cos_theta*d.x() - sin_theta*d.y(),
            sin_theta*d.x() + cos_theta*d.y(),
            d.z()
        );
        vec dir = r.direction();
        vec local_d(
            cos_theta*dir.x() - sin_theta*dir.y(),
            sin_theta*dir.x() + cos_theta*dir.y(),
            dir.z()
        );

        ray local_ray(local_o + pivot, local_d);
        if(!obj->hit(local_ray, t_min, t_max, rec, local_state)) return false;

        vec p = rec.p - pivot;
        vec n = rec.normal;
        vec world_p(
            cos_theta*p.x() + sin_theta*p.y(),
            -sin_theta*p.x() + cos_theta*p.y(),
            p.z()
        );
        vec world_n(
            cos_theta*n.x() + sin_theta*n.y(),
            -sin_theta*n.x() + cos_theta*n.y(),
            n.z()
        );

        rec.p = world_p + pivot;
        rec.normal = world_n;
        return true;
    }

    __host__ __device__ virtual bool bounding_box(aabb& output_box) const override {
        aabb local_box;
        if(!obj->bounding_box(local_box)) return false;

        vec mn( 1e30f,  1e30f,  1e30f);
        vec mx(-1e30f, -1e30f, -1e30f);

        for(int i = 0; i < 2; i++){
            for(int j = 0; j < 2; j++){
                for(int k = 0; k < 2; k++){
                    float x = i ? local_box.max().x() : local_box.min().x();
                    float y = j ? local_box.max().y() : local_box.min().y();
                    float z = k ? local_box.max().z() : local_box.min().z();

                    vec p = vec(x,y,z) - pivot;
                    vec world_corner = pivot + vec(
                        cos_theta*p.x() + sin_theta*p.y(),
                        -sin_theta*p.x() + cos_theta*p.y(),
                        p.z()
                    );

                    mn = vec(fminf(mn.x(), world_corner.x()), fminf(mn.y(), world_corner.y()), fminf(mn.z(), world_corner.z()));
                    mx = vec(fmaxf(mx.x(), world_corner.x()), fmaxf(mx.y(), world_corner.y()), fmaxf(mx.z(), world_corner.z()));
                }
            }
        }
        output_box = aabb(mn, mx);
        return true;
    }
};

__device__ vec ray_color(const ray& r , hittable** world , curandState* local_state , const vec& light_centre , float light_radius , vec light_emission){
    vec curr_attenuated(1.0f,1.0f,1.0f);
    vec accumulated_light(0.0f,0.0f,0.0f);

    ray curr_ray = r;
    float light_area = 4.0f * 3.14159265359f * light_radius * light_radius;
    float light_pdf_area = 1.0f / light_area;
    bool add_emission = true;

    const int max_depth = 30;
    const int rr_start_depth = 4;

    for(int depth = 0 ; depth < max_depth ; depth++){
        hit_record rec;
        if((*world)->hit(curr_ray,0.001f , 100000.0f , rec , local_state)){

            if(add_emission){
                accumulated_light += curr_attenuated * rec.mat->emitted();
            }

            if(rec.mat->uses_pdf_sampling()){
                cosine_pdf cos_pdf(rec.normal);
                sphere_pdf  lgt_pdf(rec.p, light_centre, light_radius);
                mixture_pdf mix_pdf(cos_pdf, lgt_pdf);

                vec scatter_dir = mix_pdf.generate(local_state);
                float pdf_val = mix_pdf.value(scatter_dir);
                if(pdf_val < 1e-6f) return accumulated_light;

                vec unit_dir = unit_vector(scatter_dir);
                float cosine = fmaxf(0.0f, dot(rec.normal, unit_dir));
                if(cosine <= 0.0f) return accumulated_light;

                vec wo = -unit_vector(curr_ray.direction());
                vec brdf = rec.mat->evaluate_brdf(unit_dir, wo, rec);

                curr_attenuated *= brdf * (cosine / pdf_val);
                curr_ray = ray(rec.p, scatter_dir);
                add_emission = true;
            } else {
                vec light_point = sample_light_point(light_centre , light_radius , local_state);

                vec to_light = light_point - rec.p;
                float distance = to_light.length();
                vec wi = to_light / distance;

                float NdotL = rec.mat->is_volumetric() ? 1.0f : fmaxf(0.0f , dot(rec.normal , wi));
                if(NdotL > 0.0f){
                    vec light_normal = unit_vector(light_point - light_centre);
                    float light_costheta = fmaxf(0.0f , dot(light_normal,-wi));

                    if(light_costheta > 0.0f){
                        ray shadow_ray(rec.p , wi);
                        hit_record shadow_rec;
                        bool blocked = (*world)->hit(shadow_ray , 0.001f , distance - 0.001f , shadow_rec , local_state);

                        if(!blocked){
                            vec wo = -unit_vector(curr_ray.direction());
                            vec brdf = rec.mat->evaluate_brdf(wi , wo , rec);

                            vec direct_light  = brdf *  light_emission * ((NdotL * light_costheta) / (distance * distance * light_pdf_area));
                            accumulated_light += curr_attenuated * direct_light;
                        }
                    }
                }
                ray scattered;
                vec attenuation;
                if(rec.mat->scatter(curr_ray, rec , attenuation , scattered , local_state)){
                    curr_attenuated *= attenuation;
                    curr_ray = scattered;
                    add_emission = rec.mat->is_delta();
                }else{
                    return accumulated_light;
                }
            }

            // Russian roulette
            if(depth > rr_start_depth){
                float survive = fmaxf(curr_attenuated.x(), fmaxf(curr_attenuated.y(), curr_attenuated.z()));
                survive = fminf(survive, 0.95f);
                if(curand_uniform(local_state) > survive){
                    break;
                }
                curr_attenuated /= survive;
            }

        }else{
            vec unit_direction = unit_vector(curr_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec sky =  (1.0f - t)  * vec(1.0f,1.0f,1.0f) + t  * vec(0.5f , 0.7f , 1.0f) ;
            accumulated_light += curr_attenuated * sky;

            return accumulated_light;
        }
    }
    return accumulated_light;
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

__global__ void render_kernel(vec* fb , int max_x , int max_y , camera cam , hittable** world , curandState* states , vec light_centre , float light_radius , vec light_emission , int samples_this_batch){
    int i = blockDim.x * blockIdx.x + threadIdx.x ;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if((i >= max_x) || (j >= max_y)) return;

    int pixel_idx = j * max_x + i;

    curandState local_state = states[pixel_idx];

    vec color_sum(0.0f, 0.0f, 0.0f);
    for(int s = 0 ; s < samples_this_batch ; s++){
        // jitter the sample within the pixel for antialiasing
        float u = (float(i) + curand_uniform(&local_state)) / float(max_x - 1);
        float v = (float(j) + curand_uniform(&local_state)) / float(max_y - 1);

        ray r = cam.get_ray(u,v,&local_state);
        color_sum += ray_color(r , world , &local_state , light_centre , light_radius , light_emission);
    }

    fb[pixel_idx] += color_sum;
    states[pixel_idx] = local_state;
}

enum MaterialType {MAT_LAMBERTIAN, MAT_METAL, MAT_DIELECTRIC , MAT_LIGHT};
struct sphereDesc{
    vec centre;
    float radius ;
    MaterialType mat_type;
    vec albedo;     // used by lambertian , metal
    float param; // roughness for metal, ir for dielectric
};

__global__ void create_world(hittable** list , hittable** world , material** materials ,sphereDesc* desc , int count , bvh_node* node_pool){
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
            case MAT_LIGHT:
                m = new emit_light(desc[i].albedo);
                break;
        }
        materials[i] = m;
        list[i] = new sphere(desc[i].centre , desc[i].radius , m);
    }

    if(count == 1){
        *world = list[0];
        return;
    }

    // Build a BVH over the same objects 
    curandState local_state;
    curand_init(1984ULL, 0, 0, &local_state);

    int pool_index = 0;
    *world = build_bvh(list, 0, count, &local_state, node_pool, &pool_index);
}
__global__ void clear_world(hittable** list, material** materials, int count) {
    if(threadIdx.x != 0 || blockIdx.x != 0) return;

    for(int i = 0; i < count; i++) {
        delete list[i];
        delete materials[i];
    }
    
}


#define NUM_EXTRA_CUBOIDS 3
#define NUM_EXTRA_TRIANGLES 2
#define NUM_EXTRA_SOLIDS (NUM_EXTRA_CUBOIDS + NUM_EXTRA_TRIANGLES)

__global__ void create_extra_geometry(
    hittable** sphere_world,        // in: sphere-BVH root
    hittable** extra_world,         // out: combined root (spheres + cuboids + triangles)
    hittable** cuboid_boundaries,   //  the raw unrotated cuboids
    material** extra_materials,     //  one material per final shape
    hittable** final_shapes,        // -- rotate wrappers (cuboids) + plain triangles
    hittable** extra_top_list       // storage, size NUM_EXTRA_SOLIDS + 1
){
    if(threadIdx.x != 0 || blockIdx.x != 0) return;

    int m = 0;


    extra_materials[m] = new lambertian(vec(0.50f, 0.46f, 0.42f));
    cuboid_boundaries[0] = new cuboid(vec(-6.1f, 0.0f, -3.6f), vec(-3.9f, 0.5f, -2.6f), extra_materials[m]);
    final_shapes[m] = new rotate_z(cuboid_boundaries[0], vec(-6.1f, 0.0f, 0.0f), 20.0f);
    m++;

    extra_materials[m] = new metal(vec(0.72f, 0.72f, 0.76f), 0.15f);
    cuboid_boundaries[1] = new cuboid(vec(4.0f, 0.0f, -2.6f), vec(5.0f, 1.0f, -1.6f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[1], vec(4.5f, 0.5f, -2.1f), 35.0f);
    m++;

    extra_materials[m] = new dielectric(1.5f);
    cuboid_boundaries[2] = new cuboid(vec(-1.7f, 0.0f, -3.9f), vec(-0.9f, 0.9f, -3.1f), extra_materials[m]);
    final_shapes[m] = new rotate_z(cuboid_boundaries[2], vec(-1.7f, 0.0f, 0.0f), 12.0f);
    m++;

    
    extra_materials[m] = new lambertian(vec(0.28f, 0.42f, 0.22f));
    final_shapes[m] = new triangle(vec(-4.20f, 1.22f, -2.60f), vec(-2.60f, 0.0f, -1.80f), vec(-2.60f, 0.0f, -3.20f), extra_materials[m]);
    m++;

  
    extra_materials[m] = new metal(vec(0.75f, 0.75f, 0.78f), 0.1f);
    final_shapes[m] = new triangle(vec(4.62f, 1.00f, -1.40f), vec(5.80f, 0.0f, -0.80f), vec(3.60f, 0.0f, -0.90f), extra_materials[m]);
    m++;

    extra_top_list[0] = *sphere_world;
    for(int i = 0; i < NUM_EXTRA_SOLIDS; i++){
        extra_top_list[i+1] = final_shapes[i];
    }

    *extra_world = new hittable_list(extra_top_list, NUM_EXTRA_SOLIDS + 1);
}

__global__ void clear_extra_geometry(hittable** extra_world, hittable** cuboid_boundaries, material** extra_materials, hittable** final_shapes){
    if(threadIdx.x != 0 || blockIdx.x != 0) return;

    for(int i = 0; i < NUM_EXTRA_SOLIDS; i++){
        delete final_shapes[i];     
        delete extra_materials[i];
    }
    for(int i = 0; i < NUM_EXTRA_CUBOIDS; i++){
        delete cuboid_boundaries[i];
    }

    // deletes the intermediate hittable_list built in create_extra_geometry
  
    delete *extra_world;
}

#define NUM_FOG_VOLUMES 6

__global__ void create_volumetrics(
    hittable** in_world,           // in: combined solids root 
    hittable** out_world,          // out: combined solids+fog root
    hittable** fog_boundaries,     
    material** fog_materials,      
    hittable** fog_media,         
    hittable** top_list,           // storage, size NUM_FOG_VOLUMES + 1
    vec hero0, vec hero1, vec hero2
){
    if(threadIdx.x != 0 || blockIdx.x != 0) return;
    int idx = 0;

    fog_boundaries[idx] = new cuboid(vec(-40.0f, -0.05f, -40.0f), vec(40.0f, 1.0f, 40.0f), nullptr);
    fog_materials[idx]  = new isotropic(vec(0.80f, 0.82f, 0.86f));
    fog_media[idx]      = new constant_medium(fog_boundaries[idx], 0.12f, fog_materials[idx]);
    idx++;

    vec heroes[3] = { hero0, hero1, hero2 };
    for(int h = 0; h < 3; h++){
        fog_boundaries[idx] = new sphere(heroes[h], 1.6f, nullptr);
        fog_materials[idx]  = new isotropic(vec(0.9f, 0.9f, 0.92f));
        fog_media[idx]      = new constant_medium(fog_boundaries[idx], 0.06f, fog_materials[idx]);
        idx++;
    }

    fog_boundaries[idx] = new sphere(vec(-4.5f, 0.6f, -3.0f), 2.3f, nullptr); // around Cuboid A + Triangle 1
    fog_materials[idx]  = new isotropic(vec(0.85f, 0.87f, 0.85f));
    fog_media[idx]      = new constant_medium(fog_boundaries[idx], 0.05f, fog_materials[idx]);
    idx++;

    fog_boundaries[idx] = new sphere(vec(4.6f, 0.6f, -1.7f), 2.3f, nullptr);  // around Cuboid B + Triangle 2
    fog_materials[idx]  = new isotropic(vec(0.88f, 0.88f, 0.90f));
    fog_media[idx]      = new constant_medium(fog_boundaries[idx], 0.05f, fog_materials[idx]);
    idx++;

    top_list[0] = *in_world;
    for(int i = 0; i < idx; i++){
        top_list[i+1] = fog_media[i];
    }

    *out_world = new hittable_list(top_list, idx + 1);
}

__global__ void clear_volumetrics(hittable** out_world, hittable** fog_boundaries, material** fog_materials, hittable** fog_media){
    if(threadIdx.x != 0 || blockIdx.x != 0) return;

    for(int i = 0; i < NUM_FOG_VOLUMES; i++){
        delete fog_media[i];
        delete fog_materials[i];
        delete fog_boundaries[i];
    }
    // Deletes the top-level hittable_list built in create_volumetrics.
    delete *out_world;
}

int main(){
    srand(42); // deterministic scene layout between runs

    const float aspect_ratio = 16.0f / 9.0f ;
    const int image_width = 800;
    const int image_height = static_cast<int>(image_width / aspect_ratio) ;

    const int total_pixels = image_height* image_width;
    const size_t fb_size = total_pixels * sizeof(vec);

    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize , 64ull*1024*1024));

    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize , 8192));

    const vec light_centre(6.0f, 6.5f, 2.0f);
    const float light_radius = 1.4f;
    const vec light_emission(9.0f, 7.0f, 5.0f); // warm key light

    std::vector<sphereDesc> h_scene;

    // ground
    h_scene.push_back({ vec(0.0f, -1000.0f, 0.0f), 1000.0f, MAT_LAMBERTIAN, vec(0.45f, 0.38f, 0.32f), 0.0f });

    // key light
    h_scene.push_back({ light_centre, light_radius, MAT_LIGHT, light_emission, 0.0f });

    std::vector<vec> accent_colors = {
        vec(2.0f,0.4f,4.0f), vec(0.3f,3.0f,3.0f), vec(4.0f,0.3f,0.6f),
        vec(0.5f,2.5f,0.8f), vec(3.0f,1.0f,0.2f), vec(0.4f,0.6f,4.0f)
    };
    for(auto& c : accent_colors){
        float x = random_range(-14.0f, 14.0f);
        float z = random_range(-18.0f, -8.0f);
        float y = random_range(1.0f, 5.0f);
        h_scene.push_back({ vec(x,y,z), 0.18f, MAT_LIGHT, c, 0.0f });
    }

    const vec hero_positions[3] = { vec(-2.4f, 1.0f, 0.0f), vec(0.0f, 1.0f, 0.0f), vec(2.4f, 1.0f, 0.0f) };
    h_scene.push_back({ hero_positions[0], 1.0f, MAT_LAMBERTIAN, vec(0.75f, 0.15f, 0.15f), 0.0f }); // warm red diffuse
    h_scene.push_back({ hero_positions[1], 1.0f, MAT_DIELECTRIC, vec(1.0f,1.0f,1.0f), 1.5f });      // glass
    h_scene.push_back({ hero_positions[2], 1.0f, MAT_METAL, vec(0.85f,0.85f,0.9f), 0.02f });        // chrome

    // field of small spheres
    for(int a = -8; a < 8; a++){
        for(int b = -8; b < 8; b++){
            float choose_mat = random_range(0.0f, 1.0f);
            vec centre(a + 0.85f*random_range(0.0f,1.0f), 0.2f, b + 0.85f*random_range(0.0f,1.0f));

            bool too_close = false;
            for(const auto& hp : hero_positions){
                vec d(centre.x()-hp.x(), centre.y()-hp.y(), centre.z()-hp.z());
                if(d.length() < 1.3f){ too_close = true; break; }
            }
            if(too_close) continue;

            if(choose_mat < 0.6f){
                vec albedo(random_range(0,1)*random_range(0,1), random_range(0,1)*random_range(0,1), random_range(0,1)*random_range(0,1));
                h_scene.push_back({ centre, 0.2f, MAT_LAMBERTIAN, albedo, 0.0f });
            } else if(choose_mat < 0.85f){
                vec albedo(random_range(0.5f,1.0f), random_range(0.5f,1.0f), random_range(0.5f,1.0f));
                float rough = random_range(0.0f, 0.4f);
                h_scene.push_back({ centre, 0.2f, MAT_METAL, albedo, rough });
            } else {
                h_scene.push_back({ centre, 0.2f, MAT_DIELECTRIC, vec(1.0f,1.0f,1.0f), 1.5f });
            }
        }
    }

    const int object_count = static_cast<int>(h_scene.size());
    std::cout << "Scene object count: " << object_count << std::endl;

    vec lookfrom(0.0f, 2.4f, 9.0f);
    vec lookat(0.1f, 1.0f, 0.0f);
    vec vup(0.0f, 1.0f, 0.0f);
    const float vfov = 35.0f;
    const float aperture = 0.08f;
    const float focus_dist = (lookfrom - vec(0.0f, 1.0f, 0.0f)).length();

    camera cam(lookfrom , lookat , vup , vfov , aspect_ratio , aperture , focus_dist);

    sphereDesc* d_descs;
    CUDA_CHECK(cudaMalloc((void**)&d_descs , object_count * sizeof(sphereDesc)));
    CUDA_CHECK(cudaMemcpy(d_descs, h_scene.data(), object_count * sizeof(sphereDesc), cudaMemcpyHostToDevice));

    vec* h_fb = (vec*)malloc(fb_size);
    vec* d_fb;
    CUDA_CHECK(cudaMalloc((void**)&d_fb , fb_size));

    curandState* d_states;
    CUDA_CHECK(cudaMalloc((void**)&d_states , total_pixels * sizeof(curandState)));

    hittable** d_list;
    CUDA_CHECK(cudaMalloc((void**)&d_list , object_count * sizeof(hittable*)));

    material** d_materials;
    CUDA_CHECK(cudaMalloc((void**)&d_materials , object_count * sizeof(material*)));

    hittable** d_world;
    CUDA_CHECK(cudaMalloc((void**)&d_world , sizeof(hittable*)));

    bvh_node* d_node_pool;
    int node_pool_count = object_count > 1 ? 2*object_count - 1 : 1;
    CUDA_CHECK(cudaMalloc((void**)&d_node_pool , node_pool_count * sizeof(bvh_node)));

    hittable** d_fog_boundaries;
    material** d_fog_materials;
    hittable** d_fog_media;
    hittable** d_top_list;
    CUDA_CHECK(cudaMalloc((void**)&d_fog_boundaries , NUM_FOG_VOLUMES * sizeof(hittable*)));
    CUDA_CHECK(cudaMalloc((void**)&d_fog_materials  , NUM_FOG_VOLUMES * sizeof(material*)));
    CUDA_CHECK(cudaMalloc((void**)&d_fog_media       , NUM_FOG_VOLUMES * sizeof(hittable*)));
    CUDA_CHECK(cudaMalloc((void**)&d_top_list        , (NUM_FOG_VOLUMES + 1) * sizeof(hittable*)));

    hittable** d_extra_world;
    hittable** d_cuboid_boundaries;
    material** d_extra_materials;
    hittable** d_final_shapes;
    hittable** d_extra_top_list;
    CUDA_CHECK(cudaMalloc((void**)&d_extra_world        , sizeof(hittable*)));
    CUDA_CHECK(cudaMalloc((void**)&d_cuboid_boundaries  , NUM_EXTRA_CUBOIDS * sizeof(hittable*)));
    CUDA_CHECK(cudaMalloc((void**)&d_extra_materials    , NUM_EXTRA_SOLIDS * sizeof(material*)));
    CUDA_CHECK(cudaMalloc((void**)&d_final_shapes       , NUM_EXTRA_SOLIDS * sizeof(hittable*)));
    CUDA_CHECK(cudaMalloc((void**)&d_extra_top_list     , (NUM_EXTRA_SOLIDS + 1) * sizeof(hittable*)));

    // final combined (solids + fog) world root -- this is  render_kernel uses
    hittable** d_final_world;
    CUDA_CHECK(cudaMalloc((void**)&d_final_world , sizeof(hittable*)));

    int tx = 8;
    int ty = 8;
    dim3 threads(tx , ty);
    dim3 blocks((image_width + tx - 1)/tx , (image_height + ty -1) / ty);

    create_world <<< 1 , 1 >>>(d_list , d_world , d_materials , d_descs , object_count , d_node_pool);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // add cuboids + triangles on top of the sphere BVH
    create_extra_geometry<<< 1 , 1 >>>(d_world , d_extra_world , d_cuboid_boundaries , d_extra_materials , d_final_shapes , d_extra_top_list);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    
    // cluster haze volumes
    create_volumetrics<<< 1 , 1 >>>(d_extra_world , d_final_world , d_fog_boundaries , d_fog_materials , d_fog_media , d_top_list , hero_positions[0] , hero_positions[1] , hero_positions[2]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    init_random_states<<< blocks, threads >>>(d_states , image_width , image_height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // multi-sample-per-pixel
    const int samples_per_pixel = 128;
    const int samples_per_batch = 16;
    const int num_batches = (samples_per_pixel + samples_per_batch - 1) / samples_per_batch;

    CUDA_CHECK(cudaMemset(d_fb, 0, fb_size));

    for(int batch = 0; batch < num_batches; batch++){
        int this_batch = std::min(samples_per_batch, samples_per_pixel - batch*samples_per_batch);
        render_kernel<<< blocks , threads >>>(d_fb , image_width , image_height , cam , d_final_world , d_states , light_centre , light_radius , light_emission , this_batch);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "Completed batch " << (batch+1) << "/" << num_batches << std::endl;
    }

    CUDA_CHECK(cudaMemcpy(h_fb , d_fb , fb_size , cudaMemcpyDeviceToHost));

    //  Write PPM
    std::ofstream file("image.ppm");
    file << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; --j) {   //start from top-left
        for (int i = 0; i < image_width; ++i) {
            int pixel_index = j * image_width + i;
            vec pixel_color = h_fb[pixel_index];

            float r_ = de_nan(pixel_color.x()) / float(samples_per_pixel);
            float g_ = de_nan(pixel_color.y()) / float(samples_per_pixel);
            float b_ = de_nan(pixel_color.z()) / float(samples_per_pixel);

            r_ = sqrtf(clamp01(r_));
            g_ = sqrtf(clamp01(g_));
            b_ = sqrtf(clamp01(b_));

            int ir = static_cast<int>(255.999f * r_);
            int ig = static_cast<int>(255.999f * g_);
            int ib = static_cast<int>(255.999f * b_);

            file << ir << " " << ig << " " << ib << "\n";
        }
    }

    file.close();

    clear_volumetrics<<< 1,1 >>>(d_final_world , d_fog_boundaries , d_fog_materials , d_fog_media);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    clear_extra_geometry<<< 1,1 >>>(d_extra_world , d_cuboid_boundaries , d_extra_materials , d_final_shapes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    clear_world <<< 1,1 >>>(d_list , d_materials, object_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_fb);
    cudaFree(d_world);
    cudaFree(d_list);
    cudaFree(d_states);
    cudaFree(d_materials);
    cudaFree(d_descs);
    cudaFree(d_node_pool);
    cudaFree(d_fog_boundaries);
    cudaFree(d_fog_materials);
    cudaFree(d_fog_media);
    cudaFree(d_top_list);
    cudaFree(d_extra_world);
    cudaFree(d_cuboid_boundaries);
    cudaFree(d_extra_materials);
    cudaFree(d_final_shapes);
    cudaFree(d_extra_top_list);
    cudaFree(d_final_world);
    free(h_fb);

    std::cout << "Successfully rendered image.ppm!" << std::endl;

    return 0;
}
