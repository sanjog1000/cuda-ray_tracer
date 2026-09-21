#ifndef PDF_H
#define PDF_H

#include "ray.h"
#include "random.h"
#include "hittable.h"
#define PI 3.14159265359f

struct onb{
    vec axis[3];

    __device__ onb(const vec& w){
        axis[2] = unit_vector(w); // local Z-axis
        vec a = (fabsf(axis[2].x()) > 0.9f) ? vec(0.0f, 1.0f, 0.0f) : vec(1.0f, 0.0f, 0.0f);  // helper func
        axis[1] = unit_vector(cross(axis[2], a));   // local Y-axis
        axis[0] = cross(axis[2], axis[1]);  // local X -axis
    }

    __device__ vec local(float x , float y , float z)const{
        return x * axis[0] + y * axis[1] + z * axis[2];
    }
    __device__ vec local(const vec&v){
        return local(v.x() ,v.y(),v.z());
    }
};

__device__ inline vec random_to_sphere(float radius, float distance_squared, curandState* local_state){
    float r1 = curand_uniform(local_state);
    float r2 = curand_uniform(local_state);

    float cos_theta_max = sqrtf(fmaxf(0.0f, 1.0f - radius * radius / distance_squared));
    float z = 1.0f + r2 * (cos_theta_max - 1.0f);

    float phi = 2.0f * PI * r1;
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float x = cosf(phi) * sin_theta;
    float y = sinf(phi) * sin_theta;

    return vec(x, y, z);
}

/* for lambertian surface light scatters in all dirns but havily biased towards surfacea_normal 
probability drops to zero as angle become parallel to surface.
formula : pdf(dir) = cos(theta)/pi */
class cosine_pdf{
public:
    onb uvw;

    __device__ cosine_pdf(const vec& normal) : uvw(normal) {};

    __device__ float value(const vec& direction)const{
        // dot product between the ray direction and the surface normal (axis[2])
        float cosine = dot(unit_vector(direction),uvw.axis[2]);
        return cosine > 0.0f ? cosine / PI : 0.0f;
    }
    __device__ vec generate(curandState* local_state){
        return uvw.local(random_cosine_direction(local_state));
    }
};

class sphere_pdf{
public: 
    vec origin;
    vec centre;
    float radius;

    __device__ sphere_pdf(const vec& orig , const vec& cen , float r) : origin(orig) , centre(cen) , radius(r) {}

    __device__ float value(const vec& direction)const{
        vec oc = centre - origin;
        float dist_sq = oc.length_squared();

        if(dist_sq <= radius* radius){
            // shading point is insode the light volume
            return 1.0f / (4.0f * PI);
        }

        float cos_theta_max = sqrtf(fmaxf(0.0f, 1.0f - (radius * radius / dist_sq)));
        float solid_angle = 2.0f * PI * (1.0f - cos_theta_max);
        if(solid_angle < 1e-8f){
            return 0.0f;
        }

    vec dir_to_centre = unit_vector(oc);
    float cos_theta = dot(unit_vector(direction) , dir_to_centre);
    
    return cos_theta >= cos_theta_max ? 1.0f / solid_angle : 0.0f ;
    }

    __device__ vec generate(curandState* local_state)const{
        vec oc = centre - origin;
        float distance_squared = oc.length_squared();
 
        if(distance_squared <= radius * radius){
            return random_unit_sphere(local_state);
        }

        onb uvw(oc);
        return uvw.local(random_to_sphere(radius , distance_squared , local_state));
    }
};

/*
if only smaple cosine pdf -- we will miss small bright lights causing high noise
if only smaple sphere pdf -- high reflective / glossy surf will lose shape & be incorrect look
so solution is combine them with 50 50
*/
class mixture_pdf{
public:
    cosine_pdf p0;
    sphere_pdf p1;

    __device__ mixture_pdf(const cosine_pdf& p0 , const sphere_pdf& p1) : p0(p0), p1(p1) {};

    __device__ float value(const vec& direction) const{
        return 0.5 * p0.value(direction) + 0.5 * p1.value(direction);
    }

    __device__ vec generate(curandState* local_state){
        // coin toss system - i) half time it generates ray pointing towards light (p1)
        // other time material dictate the direction
        if(curand_uniform(local_state) < 0.5f){
            return p0.generate(local_state);
        }else{
            return p1.generate(local_state);
        }
    }
};

class multi_sphere_pdf{
public:
    vec origin ;
    const vec* light_centres;
    const float* light_radii;
    int light_count;

    __device__ multi_sphere_pdf(const vec& orig,const vec* centres,const float* radii,int count) : origin(orig) , light_centres(centres) , light_radii(radii) , light_count(count) {}

    __device__ float value(const vec& direction)const{
        if(light_count <= 0){
            return 0.0f;
        }
        // every light is selected with probability 1 / N.
        float selection_probability = 1.0f / float(light_count);
        float result = 0.0f;

        for(int i = 0 ; i < light_count ; i++){
            sphere_pdf light_pdf(origin , light_centres[i] , light_radii[i]);

            // P(select light i) * PDF(direction | light i)
            result += selection_probability * light_pdf.value(direction);
        }
        return result;
    }
    __device__ vec generate(curandState* local_state){
        if(light_count <= 0){
            return random_unit_sphere(local_state);
        }

        int light_index = int(curand_uniform(local_state) * float(light_count));
        //  if random number lands exactly at the upper boundary.
        if(light_index >= light_count){
            light_index = light_count - 1;
        }

        sphere_pdf selected_light(origin , light_centres[light_index] , light_radii[light_index]);
        return selected_light.generate(local_state);
    }
};

class multi_mixture_pdf{
public:
    cosine_pdf cosine_part;
    multi_sphere_pdf light_part;

    __device__ multi_mixture_pdf(const cosine_pdf& cosine, const multi_sphere_pdf& lights) : cosine_part(cosine), light_part(lights) {}
    __device__ float value(const vec& direction) const{
        return 0.5f * cosine_part.value(direction) + 0.5f * light_part.value(direction);
    }
    __device__ vec generate(curandState* local_state){
        // 50% -> BRDF/cosine sampling
        if(curand_uniform(local_state) <0.5f){
            return cosine_part.generate(local_state);
        }

        // 50%-> randomly selected spherical light
        return light_part.generate(local_state);
    }
};
#endif