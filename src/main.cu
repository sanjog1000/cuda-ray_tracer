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
#include "triangle.h"
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

    // Tracks whether emission on the *next* hit still needs to be added.
    // - true at depth 0 (camera can see a light directly; nothing has
    //   accounted for that yet).
    // - true right after a pdf-sampled (lambertian) bounce, because that
    //   path relies on *implicitly* hitting the light rather than an
    //   explicit shadow ray (see the pdf-sampling branch below).
    // - false right after an explicit-NEE bounce (metal), because that
    //   branch already added the light's contribution via a shadow ray;
    //   adding it again on a direct hit would double-count it.
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
                add_emission = rec.mat->is_delta();
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
                    add_emission = false;
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
            // Dark neutral exterior so the industrial interior remains the visual subject.
            vec sky =  (1.0f - t)  * vec(0.012f, 0.016f, 0.024f) +
                       t * vec(0.055f, 0.080f, 0.120f);
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

    // Original code built a flat hittable_list here, which is O(count) per
    // ray -- fine for two spheres, very much not fine for a few hundred.
    // Build a BVH over the same objects instead (see bvh_node.h).
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
    // bvh_node internal nodes live in a pre-allocated pool (d_node_pool in
    // main()) and were never heap-allocated with `new` -- they must NOT be
    // deleted here. The pool itself is freed in one shot via cudaFree.
}

// ---------------------------------------------------------------------
// Scene geometry: a purpose-built industrial / sci-fi atrium.
//
// The final showcase scene is deliberately architectural: large cuboids and
// explicit triangle assemblies carry the composition, while GGX metals, dielectric
// glass, emissive neon surfaces, six low-density volumes, reflective flooring and
// controlled depth-of-field demonstrate the renderer's full feature set.
// There is no random RTIOW sphere field; spheres are deliberate secondary props.
// ---------------------------------------------------------------------
#define NUM_EXTRA_CUBOIDS 55
#define NUM_EXTRA_TRIANGLES 18
#define NUM_EXTRA_SOLIDS (NUM_EXTRA_CUBOIDS + NUM_EXTRA_TRIANGLES)

__global__ void create_extra_geometry(
    hittable** sphere_world,
    hittable** extra_world,
    hittable** cuboid_boundaries,
    material** extra_materials,
    hittable** final_shapes,
    hittable** extra_top_list
){
    if(threadIdx.x != 0 || blockIdx.x != 0) return;

    int m = 0;
    int c = 0;

    // -----------------------------------------------------------------
    // ROOM SHELL
    // -----------------------------------------------------------------

    // 1. Reflective floor.
    extra_materials[m] = new metal(vec(0.48f, 0.51f, 0.58f), 0.13f);
    cuboid_boundaries[c] = new cuboid(
        vec(-10.0f, 0.0f, -20.0f), vec(10.0f, 0.22f, 2.0f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(0.0f, 0.0f, 0.0f), 0.0f);
    ++m; ++c;

    // 2. Dark concrete ceiling.
    extra_materials[m] = new lambertian(vec(0.15f, 0.17f, 0.20f));
    cuboid_boundaries[c] = new cuboid(
        vec(-10.0f, 7.70f, -20.0f), vec(10.0f, 8.0f, 2.0f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(0.0f, 7.7f, 0.0f), 0.0f);
    ++m; ++c;

    // 3. Back wall.
    extra_materials[m] = new lambertian(vec(0.29f, 0.31f, 0.35f));
    cuboid_boundaries[c] = new cuboid(
        vec(-10.0f, 0.0f, -20.0f), vec(10.0f, 7.7f, -19.55f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(0.0f, 0.0f, -19.55f), 0.0f);
    ++m; ++c;

    // 4. Left wall.
    extra_materials[m] = new lambertian(vec(0.24f, 0.26f, 0.30f));
    cuboid_boundaries[c] = new cuboid(
        vec(-10.0f, 0.0f, -20.0f), vec(-9.55f, 7.7f, 2.0f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(-9.55f, 0.0f, 0.0f), 0.0f);
    ++m; ++c;

    // 5. Right wall.
    extra_materials[m] = new lambertian(vec(0.24f, 0.26f, 0.30f));
    cuboid_boundaries[c] = new cuboid(
        vec(9.55f, 0.0f, -20.0f), vec(10.0f, 7.7f, 2.0f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(9.55f, 0.0f, 0.0f), 0.0f);
    ++m; ++c;

    // -----------------------------------------------------------------
    // STRUCTURAL PILLARS / BEAMS
    // -----------------------------------------------------------------

    // 6-9. Four dark metal structural pillars.
    const vec pillar_centres[4] = {
        vec(-7.8f, 3.7f, -15.8f),
        vec( 7.8f, 3.7f, -15.8f),
        vec(-7.8f, 3.7f,  -4.2f),
        vec( 7.8f, 3.7f,  -4.2f)
    };
    for(int i = 0; i < 4; ++i){
        extra_materials[m] = new metal(vec(0.12f, 0.14f, 0.18f), 0.22f);
        vec p = pillar_centres[i];
        cuboid_boundaries[c] = new cuboid(
            vec(p.x()-0.34f, 0.0f, p.z()-0.34f),
            vec(p.x()+0.34f, 7.45f, p.z()+0.34f),
            extra_materials[m]);
        final_shapes[m] = new rotate_y(cuboid_boundaries[c], p, 0.0f);
        ++m; ++c;
    }

    // 10-12. Overhead cross-beams.
    const float beam_z[3] = { -4.5f, -10.0f, -15.5f };
    for(int i = 0; i < 3; ++i){
        extra_materials[m] = new metal(vec(0.16f, 0.18f, 0.22f), 0.20f);
        cuboid_boundaries[c] = new cuboid(
            vec(-8.9f, 6.85f, beam_z[i]-0.22f),
            vec( 8.9f, 7.25f, beam_z[i]+0.22f),
            extra_materials[m]);
        final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(0.0f, 7.05f, beam_z[i]), 0.0f);
        ++m; ++c;
    }

    // -----------------------------------------------------------------
    // CATWALKS + STAIRS
    // -----------------------------------------------------------------

    // 13-14. Raised side platforms.
    extra_materials[m] = new metal(vec(0.24f, 0.27f, 0.30f), 0.28f);
    cuboid_boundaries[c] = new cuboid(
        vec(-9.0f, 2.35f, -13.0f), vec(-5.3f, 2.65f, -7.0f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(-7.15f, 2.5f, -10.0f), 0.0f);
    ++m; ++c;

    extra_materials[m] = new metal(vec(0.24f, 0.27f, 0.30f), 0.28f);
    cuboid_boundaries[c] = new cuboid(
        vec(5.3f, 2.05f, -13.0f), vec(9.0f, 2.35f, -7.0f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(7.15f, 2.2f, -10.0f), 0.0f);
    ++m; ++c;

    // 15-19. Five-step staircase on the left, deliberately geometric rather
    // than random so the room reads like an engineered environment.
    const float step_y0[5] = {0.22f, 0.55f, 0.88f, 1.21f, 1.54f};
    const float step_z0[5] = {-4.8f, -5.8f, -6.8f, -7.8f, -8.8f};
    for(int i = 0; i < 5; ++i){
        extra_materials[m] = new lambertian(vec(0.34f, 0.36f, 0.40f));
        cuboid_boundaries[c] = new cuboid(
            vec(-8.0f, step_y0[i], step_z0[i]),
            vec(-4.7f, step_y0[i] + 0.30f, step_z0[i] + 1.10f),
            extra_materials[m]);
        final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(-6.35f, step_y0[i], step_z0[i]), 0.0f);
        ++m; ++c;
    }

    // -----------------------------------------------------------------
    // STAIR RAILS + REPEATING SUPPORTS
    // -----------------------------------------------------------------
    // Final-scene refinement: make the staircase read as engineered
    // architecture rather than a stack of boxes.  Four vertical handrail
    // posts + two long rails + two platform support columns.

    const vec rail_x[2] = { -7.86f, -4.86f };
    const float rail_z[2] = { -5.25f, -7.55f };
    for(int side = 0; side < 2; ++side){
        for(int i = 0; i < 2; ++i){
            extra_materials[m] = new metal(vec(0.56f, 0.60f, 0.66f), 0.16f);
            float x = rail_x[side];
            float z = rail_z[i];
            cuboid_boundaries[c] = new cuboid(
                vec(x-0.09f, 0.65f, z-0.09f),
                vec(x+0.09f, 2.65f, z+0.09f),
                extra_materials[m]);
            final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(x, 1.65f, z), 0.0f);
            ++m; ++c;
        }

        extra_materials[m] = new metal(vec(0.68f, 0.72f, 0.80f), 0.12f);
        float x = rail_x[side];
        cuboid_boundaries[c] = new cuboid(
            vec(x-0.09f, 2.55f, -8.90f),
            vec(x+0.09f, 2.74f, -4.75f),
            extra_materials[m]);
        final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(x, 2.64f, -6.83f), 0.0f);
        ++m; ++c;
    }

    // Two repeated vertical supports below the catwalks.
    const float support_x[2] = { -7.2f, 7.2f };
    for(int i = 0; i < 2; ++i){
        extra_materials[m] = new metal(vec(0.16f, 0.18f, 0.22f), 0.22f);
        float x = support_x[i];
        cuboid_boundaries[c] = new cuboid(
            vec(x-0.22f, 0.0f, -10.8f),
            vec(x+0.22f, 2.30f, -10.38f),
            extra_materials[m]);
        final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(x, 1.15f, -10.59f), 0.0f);
        ++m; ++c;
    }

    // -----------------------------------------------------------------
    // WALL PANELS / SHELVES
    // -----------------------------------------------------------------

    // 20-21. Back wall shelves.
    for(int i = 0; i < 2; ++i){
        extra_materials[m] = new metal(vec(0.30f, 0.32f, 0.36f), 0.16f);
        float y = 2.25f + 2.25f * i;
        cuboid_boundaries[c] = new cuboid(
            vec(-7.0f, y, -19.25f), vec(7.0f, y+0.18f, -18.75f), extra_materials[m]);
        final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(0.0f, y, -19.0f), 0.0f);
        ++m; ++c;
    }

    // 22-23. Side-wall architectural panels.
    extra_materials[m] = new metal(vec(0.11f, 0.13f, 0.16f), 0.18f);
    cuboid_boundaries[c] = new cuboid(
        vec(-9.25f, 1.5f, -15.8f), vec(-9.0f, 5.8f, -11.0f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(-9.1f, 3.6f, -13.4f), 0.0f);
    ++m; ++c;

    extra_materials[m] = new metal(vec(0.11f, 0.13f, 0.16f), 0.18f);
    cuboid_boundaries[c] = new cuboid(
        vec(9.0f, 1.5f, -15.8f), vec(9.25f, 5.8f, -11.0f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(9.1f, 3.6f, -13.4f), 0.0f);
    ++m; ++c;

    // 24-25. Ceiling inset frames.
    for(int side = -1; side <= 1; side += 2){
        extra_materials[m] = new metal(vec(0.28f, 0.30f, 0.34f), 0.10f);
        float x0 = (side < 0) ? -8.0f : 2.2f;
        float x1 = (side < 0) ? -2.2f : 8.0f;
        cuboid_boundaries[c] = new cuboid(
            vec(x0, 7.45f, -12.5f), vec(x1, 7.62f, -5.0f), extra_materials[m]);
        final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec((x0+x1)*0.5f, 7.53f, -8.75f), 0.0f);
        ++m; ++c;
    }

    // -----------------------------------------------------------------
    // CENTRAL HERO PLATFORM + FRAME
    // -----------------------------------------------------------------

    // 26-27. Central raised platform for the chrome orb / glass prism.
    extra_materials[m] = new metal(vec(0.18f, 0.20f, 0.24f), 0.20f);
    cuboid_boundaries[c] = new cuboid(
        vec(-4.8f, 0.22f, -10.1f), vec(4.8f, 0.52f, -6.1f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(0.0f, 0.37f, -8.1f), 0.0f);
    ++m; ++c;

    extra_materials[m] = new metal(vec(0.48f, 0.50f, 0.55f), 0.11f);
    cuboid_boundaries[c] = new cuboid(
        vec(-4.3f, 0.52f, -9.7f), vec(4.3f, 0.68f, -6.5f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(0.0f, 0.60f, -8.1f), 0.0f);
    ++m; ++c;

    // 28-31. A rigid frame around the central reflective orb.
    extra_materials[m] = new metal(vec(0.76f, 0.80f, 0.86f), 0.055f);
    cuboid_boundaries[c] = new cuboid(
        vec(-2.18f, 0.70f, -9.35f), vec(-1.84f, 3.88f, -8.92f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(-2.01f, 2.29f, -9.14f), 0.0f);
    ++m; ++c;

    extra_materials[m] = new metal(vec(0.76f, 0.80f, 0.86f), 0.055f);
    cuboid_boundaries[c] = new cuboid(
        vec(1.84f, 0.70f, -9.35f), vec(2.18f, 3.88f, -8.92f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(2.01f, 2.29f, -9.14f), 0.0f);
    ++m; ++c;

    extra_materials[m] = new metal(vec(0.76f, 0.80f, 0.86f), 0.055f);
    cuboid_boundaries[c] = new cuboid(
        vec(-2.18f, 3.58f, -9.35f), vec(2.18f, 3.91f, -8.92f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(0.0f, 3.75f, -9.14f), 0.0f);
    ++m; ++c;

    extra_materials[m] = new metal(vec(0.76f, 0.80f, 0.86f), 0.055f);
    cuboid_boundaries[c] = new cuboid(
        vec(-2.30f, 0.70f, -9.55f), vec(2.30f, 0.98f, -9.08f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(0.0f, 0.84f, -9.32f), 0.0f);
    ++m; ++c;

    // -----------------------------------------------------------------
    // METALLIC SCULPTURE / INDUSTRIAL MACHINERY
    // -----------------------------------------------------------------

    // 32-33. Two diagonal braces around the hero zone.
    extra_materials[m] = new metal(vec(0.52f, 0.55f, 0.60f), 0.13f);
    cuboid_boundaries[c] = new cuboid(
        vec(2.5f, 0.78f, -9.5f), vec(3.0f, 1.15f, -6.6f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(2.75f, 0.96f, -8.05f), -28.0f);
    ++m; ++c;

    extra_materials[m] = new metal(vec(0.52f, 0.55f, 0.60f), 0.13f);
    cuboid_boundaries[c] = new cuboid(
        vec(-3.0f, 0.78f, -9.5f), vec(-2.5f, 1.15f, -6.6f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(-2.75f, 0.96f, -8.05f), 28.0f);
    ++m; ++c;

    // 34-35. Tilted plates. rotate_z's sign convention here is chosen so
    // the outer edge rises rather than sinks into the floor.
    extra_materials[m] = new metal(vec(0.42f, 0.45f, 0.50f), 0.25f);
    cuboid_boundaries[c] = new cuboid(
        vec(-6.2f, 0.72f, -9.2f), vec(-3.9f, 1.05f, -7.6f), extra_materials[m]);
    final_shapes[m] = new rotate_z(cuboid_boundaries[c], vec(-6.2f, 0.72f, -9.2f), -18.0f);
    ++m; ++c;

    extra_materials[m] = new metal(vec(0.42f, 0.45f, 0.50f), 0.25f);
    cuboid_boundaries[c] = new cuboid(
        vec(4.0f, 0.72f, -9.2f), vec(6.3f, 1.05f, -7.6f), extra_materials[m]);
    final_shapes[m] = new rotate_z(cuboid_boundaries[c], vec(4.0f, 0.72f, -9.2f), 18.0f);
    ++m; ++c;

    // 36-42. A compact industrial machine housing on the right side.
    extra_materials[m] = new metal(vec(0.10f, 0.12f, 0.15f), 0.18f);
    cuboid_boundaries[c] = new cuboid(
        vec(4.9f, 0.25f, -12.7f), vec(7.7f, 1.25f, -10.6f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(6.3f, 0.75f, -11.65f), 0.0f);
    ++m; ++c;

    extra_materials[m] = new metal(vec(0.20f, 0.22f, 0.26f), 0.12f);
    cuboid_boundaries[c] = new cuboid(
        vec(5.25f, 1.25f, -12.35f), vec(5.7f, 4.1f, -11.8f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(5.48f, 2.7f, -12.1f), 0.0f);
    ++m; ++c;

    extra_materials[m] = new metal(vec(0.58f, 0.18f, 0.08f), 0.14f);
    cuboid_boundaries[c] = new cuboid(
        vec(5.0f, 3.8f, -12.55f), vec(7.7f, 4.2f, -11.0f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(6.35f, 4.0f, -11.8f), 0.0f);
    ++m; ++c;

    extra_materials[m] = new metal(vec(0.06f, 0.34f, 0.52f), 0.13f);
    cuboid_boundaries[c] = new cuboid(
        vec(7.15f, 1.2f, -12.3f), vec(7.65f, 3.8f, -11.8f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(7.4f, 2.5f, -12.05f), 0.0f);
    ++m; ++c;

    extra_materials[m] = new metal(vec(0.14f, 0.16f, 0.19f), 0.15f);
    cuboid_boundaries[c] = new cuboid(
        vec(5.05f, 1.2f, -11.2f), vec(5.55f, 3.8f, -10.7f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(5.3f, 2.5f, -10.95f), 0.0f);
    ++m; ++c;

    extra_materials[m] = new metal(vec(0.38f, 0.40f, 0.45f), 0.10f);
    cuboid_boundaries[c] = new cuboid(
        vec(4.9f, 0.95f, -10.75f), vec(7.75f, 1.3f, -10.35f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(6.32f, 1.1f, -10.55f), 0.0f);
    ++m; ++c;

    extra_materials[m] = new metal(vec(0.30f, 0.32f, 0.36f), 0.12f);
    cuboid_boundaries[c] = new cuboid(
        vec(-7.1f, 0.55f, -16.9f), vec(-4.4f, 1.35f, -15.3f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(-5.75f, 0.95f, -16.1f), 0.0f);
    ++m; ++c;

    // 43-47. Emissive architectural lighting. These are intentionally
    // broad surfaces visible to the camera; the actual NEE sampler still
    // uses the central spherical key light passed to ray_color().
    extra_materials[m] = new emit_light(vec(8.0f, 9.0f, 11.0f));
    cuboid_boundaries[c] = new cuboid(
        vec(-3.2f, 7.30f, -9.8f), vec(3.2f, 7.48f, -6.3f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(0.0f, 7.39f, -8.05f), 0.0f);
    ++m; ++c;

    extra_materials[m] = new emit_light(vec(7.0f, 0.22f, 0.10f));
    cuboid_boundaries[c] = new cuboid(
        vec(-7.2f, 5.35f, -19.05f), vec(-2.4f, 5.55f, -18.86f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(-4.75f, 5.45f, -18.96f), 0.0f);
    ++m; ++c;

    extra_materials[m] = new emit_light(vec(0.10f, 2.6f, 7.0f));
    cuboid_boundaries[c] = new cuboid(
        vec(2.4f, 5.35f, -19.05f), vec(7.2f, 5.55f, -18.86f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(4.75f, 5.45f, -18.96f), 0.0f);
    ++m; ++c;

    extra_materials[m] = new emit_light(vec(5.8f, 1.8f, 0.12f));
    cuboid_boundaries[c] = new cuboid(
        vec(-8.9f, 3.3f, -19.1f), vec(-8.72f, 5.8f, -18.85f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(-8.81f, 4.55f, -18.98f), 0.0f);
    ++m; ++c;

    extra_materials[m] = new emit_light(vec(0.18f, 2.4f, 7.0f));
    cuboid_boundaries[c] = new cuboid(
        vec(8.72f, 3.3f, -19.1f), vec(8.9f, 5.8f, -18.85f), extra_materials[m]);
    final_shapes[m] = new rotate_y(cuboid_boundaries[c], vec(8.81f, 4.55f, -18.98f), 0.0f);
    ++m; ++c;

    // -----------------------------------------------------------------
    // TRIANGLE HERO ASSEMBLIES
    // -----------------------------------------------------------------
    // Eight dielectric triangles form the main glass prism.  It is slightly
    // taller and more isolated than the earlier layout so the arbitrary
    // triangle geometry is readable as a deliberate object.
    const vec A(-5.00f, 0.80f, -9.05f);
    const vec B(-3.00f, 0.80f, -9.05f);
    const vec C(-4.00f, 3.55f, -9.05f);
    const vec D(-5.00f, 0.80f, -7.35f);
    const vec E(-3.00f, 0.80f, -7.35f);
    const vec F(-4.00f, 3.55f, -7.35f);

    extra_materials[m] = new dielectric(1.50f);
    final_shapes[m] = new triangle(A, B, C, extra_materials[m]); ++m;
    extra_materials[m] = new dielectric(1.50f);
    final_shapes[m] = new triangle(D, F, E, extra_materials[m]); ++m;
    extra_materials[m] = new dielectric(1.50f);
    final_shapes[m] = new triangle(A, D, E, extra_materials[m]); ++m;
    extra_materials[m] = new dielectric(1.50f);
    final_shapes[m] = new triangle(A, E, B, extra_materials[m]); ++m;
    extra_materials[m] = new dielectric(1.50f);
    final_shapes[m] = new triangle(B, E, F, extra_materials[m]); ++m;
    extra_materials[m] = new dielectric(1.50f);
    final_shapes[m] = new triangle(B, F, C, extra_materials[m]); ++m;
    extra_materials[m] = new dielectric(1.50f);
    final_shapes[m] = new triangle(C, F, D, extra_materials[m]); ++m;
    extra_materials[m] = new dielectric(1.50f);
    final_shapes[m] = new triangle(C, D, A, extra_materials[m]); ++m;

    // Four dielectric triangles form a larger tetrahedral crystal.  A small
    // separation between the silhouette features and the stronger colored
    // metal sculpture behind it keeps the four facets visually distinct.
    const vec T(3.85f, 4.10f, -8.40f);
    const vec L(2.55f, 1.45f, -7.15f);
    const vec R(5.15f, 1.45f, -7.15f);
    const vec Q(3.85f, 1.05f, -9.55f);

    extra_materials[m] = new dielectric(1.47f);
    final_shapes[m] = new triangle(T, L, R, extra_materials[m]); ++m;
    extra_materials[m] = new dielectric(1.47f);
    final_shapes[m] = new triangle(T, R, Q, extra_materials[m]); ++m;
    extra_materials[m] = new dielectric(1.47f);
    final_shapes[m] = new triangle(T, Q, L, extra_materials[m]); ++m;
    extra_materials[m] = new dielectric(1.47f);
    final_shapes[m] = new triangle(L, Q, R, extra_materials[m]); ++m;

    // Six metallic triangular panels make a dedicated triangle sculpture
    // beside/behind the dielectric crystal.  They are intentionally
    // differently positioned so the triangle milestone is visually obvious.
    const vec M0(5.55f, 1.35f, -7.85f);
    const vec M1(6.85f, 1.35f, -7.85f);
    const vec M2(6.20f, 3.45f, -7.85f);
    const vec N0(6.75f, 2.00f, -7.50f);
    const vec N1(7.95f, 2.00f, -7.50f);
    const vec N2(7.35f, 4.10f, -7.50f);
    const vec P0(5.25f, 2.85f, -7.15f);
    const vec P1(6.40f, 2.85f, -7.15f);
    const vec P2(5.82f, 4.65f, -7.15f);

    const vec tri_metal(0.72f, 0.76f, 0.84f);
    extra_materials[m] = new metal(tri_metal, 0.10f);
    final_shapes[m] = new triangle(M0, M1, M2, extra_materials[m]); ++m;
    extra_materials[m] = new metal(tri_metal, 0.15f);
    final_shapes[m] = new triangle(N0, N1, N2, extra_materials[m]); ++m;
    extra_materials[m] = new metal(tri_metal, 0.12f);
    final_shapes[m] = new triangle(P0, P1, P2, extra_materials[m]); ++m;

    const vec M3(5.55f, 1.35f, -7.62f);
    const vec M4(6.85f, 1.35f, -7.62f);
    const vec M5(6.20f, 3.45f, -7.62f);
    extra_materials[m] = new metal(vec(0.46f, 0.50f, 0.58f), 0.18f);
    final_shapes[m] = new triangle(M5, M4, M3, extra_materials[m]); ++m;

    const vec N3(6.75f, 2.00f, -7.28f);
    const vec N4(7.95f, 2.00f, -7.28f);
    const vec N5(7.35f, 4.10f, -7.28f);
    extra_materials[m] = new metal(vec(0.58f, 0.62f, 0.70f), 0.16f);
    final_shapes[m] = new triangle(N5, N4, N3, extra_materials[m]); ++m;

    const vec P3(5.25f, 2.85f, -6.92f);
    const vec P4(6.40f, 2.85f, -6.92f);
    const vec P5(5.82f, 4.65f, -6.92f);
    extra_materials[m] = new metal(vec(0.42f, 0.46f, 0.54f), 0.20f);
    final_shapes[m] = new triangle(P5, P4, P3, extra_materials[m]); ++m;

    // Guard against accidental scene-edit count mismatches.
    if(m != NUM_EXTRA_SOLIDS || c != NUM_EXTRA_CUBOIDS){
        printf("Scene geometry count mismatch: materials/shapes=%d expected=%d, cuboids=%d expected=%d\n",
               m, NUM_EXTRA_SOLIDS, c, NUM_EXTRA_CUBOIDS);
        return;
    }

    // Sphere BVH is one entry in the outer list; all architectural shapes
    // remain directly accessible as a small flat additive layer for now.
    extra_top_list[0] = *sphere_world;
    for(int i = 0; i < NUM_EXTRA_SOLIDS; ++i){
        extra_top_list[i+1] = final_shapes[i];
    }

    *extra_world = new hittable_list(extra_top_list, NUM_EXTRA_SOLIDS + 1);
}

__global__ void clear_extra_geometry(
    hittable** extra_world,
    hittable** cuboid_boundaries,
    material** extra_materials,
    hittable** final_shapes
){
    if(threadIdx.x != 0 || blockIdx.x != 0) return;

    for(int i = 0; i < NUM_EXTRA_SOLIDS; ++i){
        delete final_shapes[i];
        delete extra_materials[i];
    }

    // Every cuboid is owned separately by its rotate_* wrapper, so the raw
    // cuboids must be deleted after the wrappers. Triangle objects have no
    // secondary ownership here and are already fully freed above.
    for(int i = 0; i < NUM_EXTRA_CUBOIDS; ++i){
        delete cuboid_boundaries[i];
    }

    delete *extra_world;
}

// ---------------------------------------------------------------------
// Purpose-built volumetrics: low-density haze, localized around the main
// geometry so the scene has atmosphere without becoming a white fog image.
// ---------------------------------------------------------------------
#define NUM_FOG_VOLUMES 6

__global__ void create_volumetrics(
    hittable** in_world,
    hittable** out_world,
    hittable** fog_boundaries,
    material** fog_materials,
    hittable** fog_media,
    hittable** top_list,
    vec hero0, vec hero1, vec hero2
){
    if(threadIdx.x != 0 || blockIdx.x != 0) return;

    int idx = 0;

    // 1. Thin floor-level haze.
    fog_boundaries[idx] = new cuboid(
        vec(-9.2f, 0.05f, -18.5f), vec(9.2f, 0.70f, -2.0f), nullptr);
    fog_materials[idx] = new isotropic(vec(0.88f, 0.90f, 0.94f));
    fog_media[idx] = new constant_medium(fog_boundaries[idx], 0.028f, fog_materials[idx]);
    ++idx;

    // 2. Mist around the glass prism.
    fog_boundaries[idx] = new sphere(hero0, 2.20f, nullptr);
    fog_materials[idx] = new isotropic(vec(0.95f, 0.84f, 0.78f));
    fog_media[idx] = new constant_medium(fog_boundaries[idx], 0.016f, fog_materials[idx]);
    ++idx;

    // 3. Mist around the central chrome hero.
    fog_boundaries[idx] = new sphere(hero1, 2.0f, nullptr);
    fog_materials[idx] = new isotropic(vec(0.90f, 0.92f, 0.96f));
    fog_media[idx] = new constant_medium(fog_boundaries[idx], 0.014f, fog_materials[idx]);
    ++idx;

    // 4. Mist around the transparent crystal.
    fog_boundaries[idx] = new sphere(hero2, 2.35f, nullptr);
    fog_materials[idx] = new isotropic(vec(0.78f, 0.89f, 1.00f));
    fog_media[idx] = new constant_medium(fog_boundaries[idx], 0.014f, fog_materials[idx]);
    ++idx;

    // 5. Haze against the back wall for visible depth / light shafts.
    fog_boundaries[idx] = new cuboid(
        vec(-8.6f, 1.0f, -18.8f), vec(8.6f, 6.8f, -17.0f), nullptr);
    fog_materials[idx] = new isotropic(vec(0.84f, 0.88f, 0.94f));
    fog_media[idx] = new constant_medium(fog_boundaries[idx], 0.010f, fog_materials[idx]);
    ++idx;

    // 6. Soft haze around the main overhead light.
    fog_boundaries[idx] = new sphere(vec(0.0f, 6.15f, -9.8f), 2.2f, nullptr);
    fog_materials[idx] = new isotropic(vec(0.86f, 0.90f, 0.96f));
    fog_media[idx] = new constant_medium(fog_boundaries[idx], 0.010f, fog_materials[idx]);
    ++idx;

    top_list[0] = *in_world;
    for(int i = 0; i < idx; ++i){
        top_list[i+1] = fog_media[i];
    }

    *out_world = new hittable_list(top_list, idx + 1);
}

__global__ void clear_volumetrics(
    hittable** out_world,
    hittable** fog_boundaries,
    material** fog_materials,
    hittable** fog_media
){
    if(threadIdx.x != 0 || blockIdx.x != 0) return;

    for(int i = 0; i < NUM_FOG_VOLUMES; ++i){
        delete fog_media[i];
        delete fog_materials[i];
        delete fog_boundaries[i];
    }

    delete *out_world;
}

int main(){
    // Final deterministic showcase scene.  All 14 final layout requirements
    // are encoded below; the scene remains entirely in main.cu and keeps the
    // existing renderer/header architecture intact.
    srand(42);

    const float aspect_ratio = 1.0f;            // 1:1 final showcase
    const int image_width = 1920;
    const int image_height = 1920;

    const int total_pixels = image_height * image_width;
    const size_t fb_size = total_pixels * sizeof(vec);

    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64ull * 1024 * 1024));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    // -----------------------------------------------------------------
    // FINAL LIGHTING DESIGN
    // -----------------------------------------------------------------
    // Central NEE-sampled key: smaller apparent emitter, slightly moved up/back,
    // with enough radiance to preserve usable illumination.
    const vec light_centre(0.0f, 6.75f, -9.80f);
    const float light_radius = 1.20f;
    const vec light_emission(24.0f, 20.0f, 17.0f);

    std::vector<sphereDesc> h_scene;

    // 6. Main white key: smaller on camera, still responsible for the bulk of
    // neutral illumination.  Colored architecture below supplies the visual identity.
    h_scene.push_back({ light_centre, light_radius, MAT_LIGHT, light_emission, 0.0f });

    // 11. Symmetric architectural accent bulbs: red/orange on the left,
    // cyan/blue on the right.  They are deterministic, not randomly scattered.
    h_scene.push_back({ vec(-7.6f, 5.6f, -15.5f), 0.26f, MAT_LIGHT, vec(5.8f, 0.10f, 0.05f), 0.0f });
    h_scene.push_back({ vec( 7.6f, 5.6f, -15.5f), 0.26f, MAT_LIGHT, vec(0.05f, 1.8f, 6.0f), 0.0f });
    h_scene.push_back({ vec(-5.8f, 4.9f, -12.5f), 0.20f, MAT_LIGHT, vec(5.0f, 1.0f, 0.06f), 0.0f });
    h_scene.push_back({ vec( 5.8f, 4.9f, -12.5f), 0.20f, MAT_LIGHT, vec(0.06f, 1.2f, 5.2f), 0.0f });

    // 1-2-4-9. Hero/supporting spheres.  The chrome orb is the main focal
    // point; the warm diffuse hero has been moved deeper and reduced so it
    // no longer dominates the right edge; the foreground glass sphere is smaller.
    const vec hero_spheres[3] = {
        vec(0.00f, 2.15f, -8.90f),     // central chrome orb
        vec(6.40f, 1.05f, -14.60f),    // warm diffuse sphere moved into background
        vec(-1.65f, 1.05f, -6.10f)     // smaller foreground glass sphere
    };

    h_scene.push_back({ hero_spheres[0], 1.12f, MAT_METAL,
                        vec(0.95f, 0.97f, 1.00f), 0.012f });
    h_scene.push_back({ hero_spheres[1], 0.66f, MAT_LAMBERTIAN,
                        vec(0.72f, 0.16f, 0.07f), 0.0f });
    h_scene.push_back({ hero_spheres[2], 0.72f, MAT_DIELECTRIC,
                        vec(1.0f, 1.0f, 1.0f), 1.50f });

    // 9-10. Deliberate small props only; no RTIOW random sphere field.
    h_scene.push_back({ vec(-7.65f, 0.74f, -5.00f), 0.42f, MAT_METAL,
                        vec(0.55f, 0.23f, 0.09f), 0.24f });
    h_scene.push_back({ vec(-5.40f, 0.62f, -12.50f), 0.32f, MAT_LAMBERTIAN,
                        vec(0.12f, 0.35f, 0.42f), 0.0f });
    h_scene.push_back({ vec(-2.00f, 0.56f, -13.50f), 0.30f, MAT_METAL,
                        vec(0.38f, 0.48f, 0.62f), 0.30f });
    h_scene.push_back({ vec( 5.80f, 0.66f, -4.50f), 0.38f, MAT_METAL,
                        vec(0.62f, 0.64f, 0.69f), 0.16f });
    h_scene.push_back({ vec( 7.30f, 0.78f, -9.80f), 0.34f, MAT_DIELECTRIC,
                        vec(1.0f, 1.0f, 1.0f), 1.50f });
    h_scene.push_back({ vec( 6.80f, 0.72f, -14.50f), 0.38f, MAT_LAMBERTIAN,
                        vec(0.08f, 0.38f, 0.18f), 0.0f });
    h_scene.push_back({ vec(-7.00f, 0.78f, -14.20f), 0.38f, MAT_METAL,
                        vec(0.45f, 0.48f, 0.53f), 0.08f });
    h_scene.push_back({ vec( 7.20f, 0.90f, -16.60f), 0.42f, MAT_METAL,
                        vec(0.28f, 0.31f, 0.36f), 0.20f });
    h_scene.push_back({ vec( 4.40f, 0.52f, -15.00f), 0.32f, MAT_LAMBERTIAN,
                        vec(0.34f, 0.12f, 0.42f), 0.0f });
    h_scene.push_back({ vec(-6.00f, 0.58f, -2.70f), 0.32f, MAT_LAMBERTIAN,
                        vec(0.14f, 0.38f, 0.17f), 0.0f });
    h_scene.push_back({ vec( 2.80f, 0.72f, -4.20f), 0.38f, MAT_DIELECTRIC,
                        vec(1.0f, 1.0f, 1.0f), 1.50f });
    h_scene.push_back({ vec(-1.60f, 0.62f, -12.00f), 0.32f, MAT_METAL,
                        vec(0.72f, 0.55f, 0.26f), 0.12f });

    const int object_count = static_cast<int>(h_scene.size());
    std::cout << "Scene sphere/light count: " << object_count << std::endl;
    std::cout << "Architectural solids: " << NUM_EXTRA_SOLIDS
              << " (" << NUM_EXTRA_CUBOIDS << " cuboids, "
              << NUM_EXTRA_TRIANGLES << " triangles)" << std::endl;

    // -----------------------------------------------------------------
    // FINAL CAMERA COMPOSITION
    // -----------------------------------------------------------------
    // 12-13. Square showcase, lower camera, ~15% farther back from the
    // original position while retaining the same target and 48-degree FOV.
    // The larger camera distance exposes more pillars/beams/stairs without
    // resorting to an exaggerated wide-angle lens.
    vec lookfrom(9.90f, 4.00f, 13.60f);
    vec lookat(0.0f, 2.60f, -8.0f);
    vec vup(0.0f, 1.0f, 0.0f);
    const float vfov = 48.0f;
    const float aperture = 0.10f;
    const float focus_dist = (lookfrom - hero_spheres[0]).length();

    camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist);

    // -----------------------------------------------------------------
    // -----------------------------------------------------------------
    // GPU MEMORY
    // -----------------------------------------------------------------
    sphereDesc* d_descs;
    CUDA_CHECK(cudaMalloc((void**)&d_descs, object_count * sizeof(sphereDesc)));
    CUDA_CHECK(cudaMemcpy(d_descs, h_scene.data(), object_count * sizeof(sphereDesc), cudaMemcpyHostToDevice));

    vec* h_fb = (vec*)malloc(fb_size);
    vec* d_fb;
    CUDA_CHECK(cudaMalloc((void**)&d_fb, fb_size));

    curandState* d_states;
    CUDA_CHECK(cudaMalloc((void**)&d_states, total_pixels * sizeof(curandState)));

    hittable** d_list;
    CUDA_CHECK(cudaMalloc((void**)&d_list, object_count * sizeof(hittable*)));

    material** d_materials;
    CUDA_CHECK(cudaMalloc((void**)&d_materials, object_count * sizeof(material*)));

    hittable** d_world;
    CUDA_CHECK(cudaMalloc((void**)&d_world, sizeof(hittable*)));

    // The sphere/light BVH contains every entry from h_scene. One bvh_node
    // exists even at a single-object leaf, hence the 2N-1 pool sizing.
    bvh_node* d_node_pool;
    int node_pool_count = object_count > 1 ? 2 * object_count - 1 : 1;
    CUDA_CHECK(cudaMalloc((void**)&d_node_pool, node_pool_count * sizeof(bvh_node)));

    // Extra architecture layer.
    hittable** d_extra_world;
    hittable** d_cuboid_boundaries;
    material** d_extra_materials;
    hittable** d_final_shapes;
    hittable** d_extra_top_list;
    CUDA_CHECK(cudaMalloc((void**)&d_extra_world, sizeof(hittable*)));
    CUDA_CHECK(cudaMalloc((void**)&d_cuboid_boundaries, NUM_EXTRA_CUBOIDS * sizeof(hittable*)));
    CUDA_CHECK(cudaMalloc((void**)&d_extra_materials, NUM_EXTRA_SOLIDS * sizeof(material*)));
    CUDA_CHECK(cudaMalloc((void**)&d_final_shapes, NUM_EXTRA_SOLIDS * sizeof(hittable*)));
    CUDA_CHECK(cudaMalloc((void**)&d_extra_top_list, (NUM_EXTRA_SOLIDS + 1) * sizeof(hittable*)));

    // Volumetric layer.
    hittable** d_fog_boundaries;
    material** d_fog_materials;
    hittable** d_fog_media;
    hittable** d_top_list;
    CUDA_CHECK(cudaMalloc((void**)&d_fog_boundaries, NUM_FOG_VOLUMES * sizeof(hittable*)));
    CUDA_CHECK(cudaMalloc((void**)&d_fog_materials, NUM_FOG_VOLUMES * sizeof(material*)));
    CUDA_CHECK(cudaMalloc((void**)&d_fog_media, NUM_FOG_VOLUMES * sizeof(hittable*)));
    CUDA_CHECK(cudaMalloc((void**)&d_top_list, (NUM_FOG_VOLUMES + 1) * sizeof(hittable*)));

    hittable** d_final_world;
    CUDA_CHECK(cudaMalloc((void**)&d_final_world, sizeof(hittable*)));

    int tx = 8;
    int ty = 8;
    dim3 threads(tx, ty);
    dim3 blocks((image_width + tx - 1) / tx, (image_height + ty - 1) / ty);

    // -----------------------------------------------------------------
    // BUILD SCENE
    // -----------------------------------------------------------------
    create_world<<<1,1>>>(d_list, d_world, d_materials, d_descs, object_count, d_node_pool);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    create_extra_geometry<<<1,1>>>(
        d_world, d_extra_world,
        d_cuboid_boundaries, d_extra_materials,
        d_final_shapes, d_extra_top_list);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // hero0 = glass prism center, hero1 = chrome orb, hero2 = transparent crystal.
    const vec hero0(-4.00f, 2.05f, -8.20f);
    const vec hero1( 0.00f, 2.15f, -8.90f);
    const vec hero2( 3.85f, 2.20f, -8.15f);

    create_volumetrics<<<1,1>>>(
        d_extra_world, d_final_world,
        d_fog_boundaries, d_fog_materials, d_fog_media,
        d_top_list, hero0, hero1, hero2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // -----------------------------------------------------------------
    // RENDER
    // -----------------------------------------------------------------
    init_random_states<<<blocks, threads>>>(d_states, image_width, image_height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int samples_per_pixel = 1024;
    const int samples_per_batch = 32;
    const int num_batches = (samples_per_pixel + samples_per_batch - 1) / samples_per_batch;

    CUDA_CHECK(cudaMemset(d_fb, 0, fb_size));

    for(int batch = 0; batch < num_batches; ++batch){
        int this_batch = std::min(
            samples_per_batch,
            samples_per_pixel - batch * samples_per_batch);

        render_kernel<<<blocks, threads>>>(
            d_fb, image_width, image_height, cam,
            d_final_world, d_states,
            light_centre, light_radius, light_emission,
            this_batch);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "Completed batch " << (batch + 1) << "/" << num_batches << std::endl;
    }

    CUDA_CHECK(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));

    // -----------------------------------------------------------------
    // OUTPUT
    // -----------------------------------------------------------------
    std::ofstream file("image.ppm");
    file << "P3\n" << image_width << " " << image_height << "\n255\n";

    for(int j = image_height - 1; j >= 0; --j){
        for(int i = 0; i < image_width; ++i){
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

    // -----------------------------------------------------------------
    // CLEANUP
    // -----------------------------------------------------------------
    clear_volumetrics<<<1,1>>>(
        d_final_world,
        d_fog_boundaries, d_fog_materials, d_fog_media);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    clear_extra_geometry<<<1,1>>>(
        d_extra_world,
        d_cuboid_boundaries, d_extra_materials, d_final_shapes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    clear_world<<<1,1>>>(d_list, d_materials, object_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_fb);
    cudaFree(d_world);
    cudaFree(d_list);
    cudaFree(d_states);
    cudaFree(d_materials);
    cudaFree(d_descs);
    cudaFree(d_node_pool);

    cudaFree(d_extra_world);
    cudaFree(d_cuboid_boundaries);
    cudaFree(d_extra_materials);
    cudaFree(d_final_shapes);
    cudaFree(d_extra_top_list);

    cudaFree(d_fog_boundaries);
    cudaFree(d_fog_materials);
    cudaFree(d_fog_media);
    cudaFree(d_top_list);
    cudaFree(d_final_world);

    free(h_fb);

    std::cout << "Successfully rendered image.ppm!" << std::endl;
    return 0;
}
