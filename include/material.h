#ifndef MATERIAL_H
#define MATERIAL_H
#include"ray.h"
#include "hittable.h"
#include "random.h"

__device__ inline vec random_unit_sphere(curandState* local_state){
    while(true){
        vec p = vec(
            random_float(-1.0f ,1.0f , local_state),
            random_float(-1.0f ,1.0f , local_state),
            random_float(-1.0f ,1.0f , local_state)
        );
        float len_sq = p.length_squared(); 
        if(len_sq >= 1.0f || len_sq < 1e-8f) continue;

        return unit_vector(p) ;
    }
}
__host__ __device__ inline vec reflect(const vec& v , const vec& n){
    return v - 2.0f * dot(v,n) * n ;  
}

__host__ __device__ inline vec refract(const vec& r_in , const vec& n , float refraction_ratio){
    float cos_theta = fmin(dot(-r_in, n), 1.0f);
    vec r_out_perp = refraction_ratio * (r_in + cos_theta * n);
    vec r_out_parallel = -sqrtf(fabsf(1.0f - (r_out_perp.length_squared()))) * n;

    return (r_out_parallel + r_out_perp) ;
}
// Probability that the surface acts like a mirror -- it is a shortcut that mimicks frensel's effect
__host__ __device__ inline float schlick(float cosine , float ref_idx){
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0*r0;
    float reflectance =  r0 + ((1 - r0) * powf((1 - cosine) , 5));
    
    return reflectance;
}
class material{
public:
     __device__ virtual bool scatter(const ray& r_in , const hit_record& rec , vec& attenuation ,ray& scattered , curandState* local_state) const = 0;
};

class lambertian : public material{
public:
    vec albedo;
    __device__ lambertian(const vec& a) : albedo(a) {}

    __device__ virtual bool scatter(const ray& r_in , const hit_record& rec , vec& attenuation ,ray& scattered , curandState* local_state) const override{
        vec scatter_dirn = rec.normal + random_unit_sphere(local_state);

        attenuation = albedo;
        scattered = ray(rec.p , scatter_dirn);

        return true;
    }
};
class metal : public material{
public:
    vec albedo;
    float fuzz;

    __device__ metal(const vec& a , float f) : albedo(a) , fuzz(f < 0.0f ? 0.0f : (f < 1.0f ? f : 1.0f) ) {}
    
    __device__ virtual bool scatter(const ray& r_in , const hit_record& rec , vec& attenuation ,ray& scattered , curandState* local_state)const override{
        // calculate the perfect reflection : r = v - 2(v.n)*n
        vec reflected = reflect(unit_vector(r_in.direction()), rec.normal);

        scattered = ray(rec.p , reflected +  fuzz * random_unit_sphere(local_state)) ;
        attenuation = albedo;

        return (dot(scattered.direction() , rec.normal) > 0);
    }
}; 

class dielectric : public material {
public:
    float ir;
    __device__ dielectric(float refractive_idx) : ir(refractive_idx) {}
    
    __device__ virtual bool scatter(const ray& r_in , const hit_record& rec , vec& attenuation ,ray& scattered ,curandState* local_state)const override{
        // glass doesn't absorb any light : only reflect or refract 
        attenuation = vec(1.0f ,1.0f ,1.0f);

        float refraction_ratio = rec.front_face ? (1.0 / ir) : ir ;
        vec unit_dirn = unit_vector(r_in.direction());

        float cos_theta = fmin(dot(-unit_dirn , rec.normal) , 1.0f);
        float sin_theta = sqrtf(1.0f - (cos_theta * cos_theta));

        // Condition : for TIR --> ref_idx * sin_theta should be more than 1
        bool cant_refract = (refraction_ratio * sin_theta) > 1.0f ;
        
        vec direction ;
        if(cant_refract || schlick(cos_theta , refraction_ratio) > random_float(local_state)){
            // acts like a mirror -- it is trapped or it schlick reflectance is more than the random guess
            direction = reflect(unit_dirn , rec.normal);
        }else{
            direction = refract(unit_dirn, rec.normal , refraction_ratio);
        }

        scattered = ray(rec.p , direction);
        return true;
    }
};

class isotropic : public material {
public:
    vec albedo;
    __device__ isotropic(vec a) : albedo(a) {}

    __device__ virtual bool scatter(const ray& r_in , const hit_record& rec , vec& attenuation ,ray& scattered , curandState* local_state) const override{
        // generate a random vector inside a 3D sphere.
        vec random_direction = random_unit_sphere(local_state);
        
        // after hitting the particle : fire a new ray 
        // starting exactly at the dust particle( rec.p) and shoot at random dirn.
        scattered = ray(rec.p , random_direction);

        attenuation = albedo;
        return true;
    }
};
#endif