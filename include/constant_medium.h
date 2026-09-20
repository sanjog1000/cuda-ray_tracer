#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H
#include "hittable.h"
#include "material.h"
#include "random.h"
#include<float.h>
class constant_medium : public hittable{
public:
    hittable* boundary ; // The glass box or sphere (shape of the container) holding the fog
    material* phase_func ; // How the light bounces when it hits a dust particle

    float neg_inv_density ; // why negative and inverse ? from beer-lambert law : p(x) = e ^ (-d.x)  => x = (- ln(p) / d )
    // so store the the density as -1/d so that we can just multiply it rather than dividing (which is expensive)

    __host__ __device__ constant_medium(hittable* b , float d , material* a) : boundary(b) , neg_inv_density(-1.0f / d) , phase_func(a) {}

    __host__ __device__ virtual bool bounding_box(aabb& output_box)const override{
        // The fog takes up the exact same space as the boundary holding it
        return boundary->bounding_box(output_box);

    }
    __host__ __device__ virtual bool hit(const ray& r , float t_Min ,float t_max , hit_record& rec , curandState* local_state) const override;
};

__host__ __device__   bool constant_medium::hit(const ray& r , float t_min ,float t_max , hit_record& rec , curandState* local_state)const {
    #if defined(__CUDA_ARCH__)
    hit_record rec1 , rec2;

    // find where the ray enters the container
    // Shoot a laser from -ve infinity to +ve infinity to find the front wall
    if(!boundary->hit(r , -FLT_MAX , FLT_MAX , rec1 , local_state)){
        return false;
    }

    // find where the ray exits the container
    // Shoot a laser , starting just past the entry point
    if(!boundary->hit(r , rec1.t+0.0001f , FLT_MAX , rec2, local_state)){
        return false;
    }

    // clamping the entry-exit to actual camera's vision limits
    if(rec1.t < t_min) rec1.t = t_min;
    if(rec2.t > t_max) rec2.t = t_max;
    if(rec1.t >= rec2.t) return false;
    if(rec1.t < 0) rec1.t = 0;

    // total distance through container = no. of steps(rec2.t - rec1.t) * length of each step
    float step_length = r.direction().length();
    float dist_inside_boundary = (rec2.t - rec1.t) * step_length;

    //we use a random no. to predict how far the ray will travel it hits any particle
    //curand_uniform - produce float btw 0 and 1
    float hit_dist = neg_inv_density * logf(curand_uniform(local_state));   // this is beer-lambert law

    // the hit happend outside the container so ray completely passed the fog without any hit
    if(hit_dist > dist_inside_boundary) return false;

    // when hit inside the fog
    rec.t = rec1.t + (hit_dist / step_length);
    rec.p = r.parametric_eqn(rec.t);

    // in a volume : there is no surface normal: so just pointing forward for the math
    rec.normal = vec(1,0,0);
    rec.front_face = true;
    rec.mat = phase_func;
    return true;

    #else
        return false;

    #endif
}
#endif
