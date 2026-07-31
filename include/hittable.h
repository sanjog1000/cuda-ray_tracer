#ifndef HITTABLE_H
#define HITTABLE_H

#include"ray.h"

class material ;    // telling the compiler the material class exists somewhere

struct hit_record{
    vec p ;
    vec normal ;
    float t;
    bool front_face;
    material* mat;

    __device__ inline void set_face_normal(const ray& r, const vec& outward_normal){
        front_face = dot(r.direction() , outward_normal) < 0.0f ;
        normal = front_face ? outward_normal : -outward_normal ;
    }
};

class hittable{
public:
    __device__ virtual bool hit(const ray& r , float t_min , float t_max , hit_record& rec) const = 0 ;    
};
#endif