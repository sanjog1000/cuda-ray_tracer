#ifndef CUBOID_H
#define CUBOID_H
#include"hittable.h"
#include"ray.h"
#include  "aabb.h"
#include "random.h"
#include<utility>
class material ;

class cuboid : public hittable{
public:
    vec min_bounds;
    vec max_bounds;
    material* mat_ptr ;

    __host__ __device__ cuboid() : min_bounds() , max_bounds() , mat_ptr(nullptr) {}
    __host__ __device__ cuboid(vec min_b , vec max_b , material* m) : min_bounds(min_b) , max_bounds(max_b) , mat_ptr(m) {}

    __host__ __device__ virtual bool hit(const ray& r , float t_min , float t_max , hit_record& rec ,curandState* local_state) const override;

    __host__ __device__ virtual bool bounding_box(aabb& output_box) const override{
        vec padding(0.0001f , 0.0001f , 0.0001f);
        output_box = aabb(min_bounds - padding , max_bounds + padding);
        return true;
    }
};

__host__ __device__ bool cuboid::hit(const ray& r , float t_min , float t_max , hit_record& rec ,curandState* local_state)const{
    float t_entry = t_min;
    float t_exit = t_max;

    vec entry_normal(0.0f,0.0f,0.0f);
    vec exit_normal(0.0f,0.0f,0.0f);

    for(int  a = 0 ; a < 3 ; a++){

        float direction = r.direction()[a];

        if(fabsf(direction) < 1e-8f){
            // origin is outside the slab so it can never hit cuboid
            if(r.origin()[a] < min_bounds[a] || r.origin()[a] > max_bounds[a]){
                return false;
            }
            // origin inside slab
            continue;
        }

        float invD = 1.0f / direction;
        float t0 = (min_bounds[a] - r.origin()[a]) * invD;
        float t1 = (max_bounds[a] - r.origin()[a]) * invD;

        vec normal0(0.0f, 0.0f, 0.0f);
        vec normal1(0.0f, 0.0f, 0.0f);

        // t0 + normal0 = ENTRY
        // t1 + normal1 = EXIT

        normal0[a] = -1.0f;   // min face outward normal
        normal1[a] =  1.0f;   // max face outward normal

        if(invD < 0.0f){
            std::swap(t0,t1);
            
            std::swap(normal0,normal1);    
        }

        // Latest entering point
        if(t0 > t_entry){
            t_entry = t0;
            entry_normal = normal0;
        }
        // Earliest exiting point
        if(t1 < t_exit){
            t_exit = t1;
            exit_normal = normal1;
        }

        if(t_exit <= t_entry) return false;
    }

    bool starts_inside = true;
    for(int a = 0 ; a < 3 ; a++){
        if(r.origin()[a] < min_bounds[a] || r.origin()[a] > max_bounds[a]){
                starts_inside = false;
                break;
        }
    }
     // Outside ray -> first surface is entry.
    // Inside ray-> first surface is exit.
    float hit_t;
    vec hit_normal;

    if(starts_inside){
        hit_t = t_exit;
        hit_normal = exit_normal;
    }else{
        hit_t = t_entry;
        hit_normal = entry_normal;
    }


    if(hit_t  < t_min || hit_t > t_max) return false;  // after confirming hit : check whether the hit took place behind camera or too far way from the camera

    rec.t = t_entry;
    rec.p = r.parametric_eqn(rec.t);
    rec.set_face_normal(r ,hit_normal);
    rec.mat = mat_ptr ;

    return true;
}
#endif
