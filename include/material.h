#ifndef MATERIAL_H
#define MATERIAL_H
#include"ray.h"
#include "hittable.h"
#include "random.h"

__device__ inline float ggx_distribution(const vec& n , const vec& h , float roughness){
    roughness = fmaxf(roughness, 0.001f);
    float alpha = roughness * roughness;
    float alpha_sq = alpha* alpha;

    float n_dot_h = fmaxf(dot(n,h),0.0f);
    float n_dot_h_sq = n_dot_h*n_dot_h;

    float denominatar = n_dot_h_sq * (alpha_sq - 1.0f) +1.0f ;
    return alpha_sq / (3.14159265359f  *  denominatar *denominatar) ;
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
    float reflectance =  r0 + ((1 - r0) * powf((1.0f - cosine) , 5));
    
    return reflectance;
}
// this is RGB version of the schilick 
__device__ inline vec metal_schilick(float cosine ,const vec& f0){
    float power_term = powf((1.0f - cosine),5);
    return f0+ (vec(1.0f,1.0f,1.0f) - f0)*power_term ;
}

__device__ inline float smith_ggx(float NdotW , float alpha){
    NdotW = fmaxf(NdotW ,0.0f);
    float NdotW_sq = NdotW*NdotW;

    float denominator = NdotW + sqrtf(NdotW_sq + alpha*alpha*(1.0f - NdotW_sq));

    return (2.0f * NdotW )/ denominator ; 
}

__device__ inline vec cook_torrance_brdf(const vec& n , const vec& wi ,const vec& wo , const vec& f0 , float roughness){
    float NdotL = fmaxf(dot(n,wi),0.0f);
    float NdotV = fmaxf(dot(n,wo),0.0f);

    if(NdotV <= 0.0f || NdotL <= 0.0f) return vec(0.0f,0.0f,0.0f);

    vec h = unit_vector(wi+wo);

    float VdotH = fmaxf(dot(wo,h),0.0f);
    roughness = fmaxf(roughness, 0.001f);
    float alpha = roughness * roughness;

    float D = ggx_distribution(n,h , roughness);

    vec F = metal_schilick(VdotH,f0);

    float G1_L = smith_ggx(NdotL,alpha);
    float G1_V = smith_ggx(NdotV,alpha);
    float G = G1_L* G1_V;

    // geometric correction factor - (4 .(n . wi).(n . wo))
    float denominator =fmaxf(4.0f * NdotL * NdotV, 1e-4f);
    return F * (D * G / denominator);
}

__device__ inline vec sample_half_vector(const vec& n , float roughness , curandState* local_state){
    roughness = fmaxf(roughness, 0.001f);
    float alpha =roughness* roughness;

    float r1 = random_float(local_state);
    float r2 = random_float(local_state);

    float phi = 2* r1 * 3.14159265359f ;
    float cos_theta = sqrtf((1.0f - r2) / (1.0f + (alpha*alpha -1.0f) * r2));
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));

    vec h_local(sin_theta * cosf(phi) , sin_theta * sinf(phi) , cos_theta);

    vec up =
        (fabsf(n.z()) < 0.999f)
        ? vec(0.0f, 0.0f, 1.0f)
        : vec(1.0f, 0.0f, 0.0f);

    vec tangent = unit_vector(cross(up,n));
    vec bitangent = cross(n,tangent);
    
    return unit_vector(tangent * h_local.x() + bitangent * h_local.y() + n* h_local.z());
}

__device__ inline float ggx_pdf(const vec& n , const vec& h, const vec& wo ,  float roughness){
    roughness = fmaxf(roughness, 0.001f);

    float NdotH = fmaxf(0.0f , dot(n,h));
    float VdotH = fmaxf(0.0f , dot(wo,h));

    if(NdotH <= 0.0f || VdotH <= 0.0f) return 0.0f;
    
    float D = ggx_distribution(n , h , roughness);
    // pddf = D.(n.h) / (4.(v.h))
    return (D * NdotH) / (4 * VdotH) ;
}
class material{
public:
    __device__ virtual bool scatter(const ray& r_in , const hit_record& rec , vec& attenuation ,ray& scattered , curandState* local_state) const = 0;
    
    __device__ virtual vec emitted()const{
        return vec(0.0f,0.0f,0.0f);
    }

    __device__ virtual vec evaluate_brdf(const vec& wi ,const vec& wo , const hit_record& rec)const{
        return vec(0.0f,0.0f,0.0f);
    }
    __device__ virtual bool uses_pdf_sampling() const { return false; }   
    __device__ virtual bool is_volumetric() const { return false; }    
    __device__ virtual bool is_delta() const {return false;}
    
    __device__ virtual ~material() {}
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

    __device__ virtual vec evaluate_brdf(const vec& wi ,const vec& wo , const hit_record& rec)const override{
        return albedo / 3.14159265359f;
    }
    
    __device__ virtual bool uses_pdf_sampling() const override { return true; }

};
 
class emit_light : public material{
public:
    vec emit_color;

    __device__ emit_light(const vec& e) : emit_color(e) {}

    __device__ bool scatter(const ray& r_in , const hit_record& rec , vec& attenuation ,ray& scattered , curandState* local_state)const override {
        return false;
    }
    __device__ vec emitted()const override{
        return emit_color;
    }
};

class metal : public material{
public:
    vec albedo;
    float roughness;

    __device__ metal(const vec& a , float r) : albedo(a) , roughness(fminf(fmaxf(r, 0.001f), 1.0f)) {}

    __device__ virtual bool scatter(const ray& r_in , const hit_record& rec , vec& attenuation , ray& scattered , curandState* local_state)const override{
        // dirn from surface toward the previous bounce
        vec wi = -unit_vector(r_in.direction());

        // sample a GGX microfacet normal
        vec h = sample_half_vector(rec.normal , roughness, local_state);

        // Reflect incoming direction around the sampled microfacet
        vec wo = reflect(-wi , h);

        float NdotV  = fmaxf(0.0f, dot(rec.normal , wo));
        if(NdotV <= 0.0f) return false;

        // Probability density of this sampled outgoing direction
        float pdf = ggx_pdf(rec.normal,h , wo ,roughness);
        if(pdf <= 1e-8f) return false;  // safety check

        // Cook-Torrance BRDF
        vec brdf = cook_torrance_brdf(rec.normal , wi,wo , albedo ,roughness);

        // Monte Carlo weight : BRDF * cos(theta) / PDF
        attenuation = brdf * (NdotV / pdf);
        scattered = ray(rec.p, wo);
        return true;
    }
    
    __device__ virtual vec evaluate_brdf(const vec& wi ,const vec& wo , const hit_record& rec)const override{
        return cook_torrance_brdf(rec.normal , wi , wo , albedo , roughness);
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

    __device__ bool is_delta() const override{return true;}
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
    __device__ virtual vec evaluate_brdf(const vec& wi ,const vec& wo , const hit_record& rec)const override{
        return albedo / (4.0f * 3.14159265359f);   // isotropic phase function
    }
    __device__ virtual bool is_volumetric() const override { return true; }
};
#endif