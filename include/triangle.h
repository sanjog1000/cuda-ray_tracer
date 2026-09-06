#ifndef TRIANGLE_H
#define TRIANGLE_H
#include "ray.h"
#include "hittable.h"
#include "aabb.h"
#include "random.h"
class material ;

class triangle : public hittable{
public:
    vec v0 , v1 , v2 ;
    material* mat_ptr;

    __host__ __device__ triangle() : v0() , v1() , v2() , mat_ptr(nullptr) {}
    __host__ __device__ triangle(vec a , vec b , vec c , material* m) : v0(a) , v1(b) ,v2(c) , mat_ptr(m) {}

    __host__ __device__ virtual bool hit(const ray& r , float t_min , float t_max , hit_record& rec , curandState* local_state) const override;

    __host__ __device__ virtual bool bounding_box(aabb& output_box)const override{
        vec min_point(fminf(v0.x() , fminf(v1.x() , v2.x())),
                    fminf(v0.y() , fminf (v1.y() , v2.y())),
                    fminf(v0.z() , fminf (v1.z() , v2.z())));

        vec max_point(fmaxf(v0.x() , fmaxf(v1.x() , v2.x())),
                    fmaxf(v0.y() , fmaxf (v1.y() , v2.y())),
                    fmaxf(v0.z() , fmaxf (v1.z() , v2.z())));

        vec padding(0.0001f , 0.0001f , 0.0001f);

        output_box = aabb(min_point - padding , max_point + padding);

        return true;
    }
};

__host__ __device__ bool triangle::hit(const ray& r , float t_min , float t_max , hit_record& rec , curandState* local_state) const{
    vec e1 = v1 - v0;
    vec e2 = v2 - v0;

    vec h = cross(r.direction(), e2);
    float a = dot(e1 , h);

    // if 'a' is extremely close to 0 --> it is || to surf and never intersects
    if(a > -0.00001f && a < 0.00001f) return false;

    float f = 1.0f / a ;

    vec s = r.origin() - v0;
    float u = f * dot(s ,h);
    if(u < 0.0f || u > 1.0f){
        return false;
    }

    vec q = cross(s , e1);
    float v = f*dot(r.direction(),q);
    if(v < 0.0f || v > 1.0f || u + v >1.0f){
        return false;
    }

    float t = f * dot(e2 , q);
    // or we can use easy apply of cramer's rule but we have to again calculate a cross product (which is costly) 
    //  t = -f * dot(e1, cross(s, e2)); 
    if(t < t_min || t > t_max){
        return false;
    }

    rec.t = t;
    rec.p = r.parametric_eqn(rec.t);
    vec outward_normal = unit_vector(cross(e1 ,e2));
    rec.set_face_normal(r , outward_normal);
    rec.mat = mat_ptr;
    
    return true;
}
#endif