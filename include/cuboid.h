#ifndef CUBOID_H
#define CUBOID_H
#include"hittable.h"
#include"ray.h"
class material ;

class cuboid : public hittable{
public:
    vec min_bounds;
    vec max_bounds;
    material* mat_ptr ;
public:
    __device__ cuboid() {}
    __device__ cuboid(vec min_b , vec max_b , material* m) : min_bounds(min_b) , max_bounds(max_b) , mat_ptr(m) {}

    __device__ virtual bool hit(const ray& r , float t_min , float t_max , hit_record& rec) const override;
};

__device__ bool cuboid::hit(const ray& r , float t_min , float t_max , hit_record& rec)const{
    float t_entry = t_min;
    float t_exit = t_max;
    
    vec outward_normal(0.0f , 0.0f , 0.0f);

    for(int  a = 0 ; a < 3 ; a++){
        float invD = 1/ r.direction()[a];
        float t0 = (min_bounds[a] - r.origin()[a]) * invD;
        float t1 = (max_bounds[a] - r.origin()[a]) * invD;

        vec surf_normal(0.0f , 0.0f , 0.0f);
        surf_normal[a] = -1.0f;

        if(invD < 0.0f){
            float temp = t0;
            t0 = t1;
            t1 = temp;
            surf_normal[a] = 1.0f;
        }

        if(t0 > t_entry){
            t_entry = t0;
            outward_normal = surf_normal;   
        }
        if(t1 < t_exit){
            t_exit = t1;
        }

        if(t_exit <= t_entry) return false;
    }
    if(t_entry  < t_min || t_entry > t_max) return false;  // after confirming hit : check whether the hit took place behind camera or too far way rrom the camera

    rec.t = t_entry;
    rec.p = r.parametric_eqn(rec.t);
    rec.set_face_normal(r ,outward_normal);

    rec.mat = mat_ptr ;
    
    return true;
}
#endif