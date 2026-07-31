#ifndef SPHERE_H
#define SPHERE_H
#include"ray.h"
#include"hittable.h"

class material ;

class sphere : public hittable{
public: 
    vec centre;
    float radius;
    material* mat_ptr;
public:
    __device__ sphere() {}
    __device__ sphere(vec cen , float r) : centre(cen), radius(r) {}

    __device__ virtual bool hit(const ray& r , float t_min , float t_max , hit_record& rec) const override;
};

__device__ bool sphere::hit(const ray& r , float t_min , float t_max , hit_record& rec) const{
    vec oc = r.origin() - centre;
    float a = dot(r.direction() , r.direction());
    float b = 2.0f * dot(r.direction() , oc );
    float c = dot(oc , oc) - radius*radius ;

    float discriminant = b*b - 4.0f *a*c ;
    if(discriminant < 0.0f) return false;

    float sqrt_root = sqrtf(discriminant);
    float root = (-b - sqrt_root) / (2.0f * a) ;

    
    if(root < t_min || root > t_max){
        root = (-b + sqrt_root) / (2.0f * a) ;
        if(root < t_min || root > t_max){
            return false;
        }
    }

    rec.t = root;
    rec.p = r.parametric_eqn(rec.t);
    vec outward_normal = (rec.p - centre) / radius;
    rec.set_face_normal(r , outward_normal);
    rec.mat = mat_ptr;
    
    return true;
}
#endif