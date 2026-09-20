#ifndef HITTABLE_H
#define HITTABLE_H

#include"ray.h"
#include "aabb.h"
#include "random.h"
class material ;    // forward declare

struct hit_record{
    vec p ;
    vec normal ;
    float t;
    bool front_face;
    material* mat;

    __host__ __device__ inline void set_face_normal(const ray& r, const vec& outward_normal){
        front_face = dot(r.direction() , outward_normal) < 0.0f ;
        normal = front_face ? outward_normal : -outward_normal ;
    }
};

class hittable{
public:
    __host__ __device__ virtual bool hit(const ray& r , float t_min , float t_max , hit_record& rec , curandState* local_state) const = 0 ;   
    
    __host__ __device__ virtual bool bounding_box(aabb& output_box)const =0;

    __host__ __device__ virtual ~hittable() {}; 
};
#endif