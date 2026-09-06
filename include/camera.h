#ifndef CAMERA_H
#define CAMERA_H
#include "ray.h"
#include "vec.h"
#include "random.h"

__device__ inline vec random_unit_disk(curandState* local_state){
    while(true){
        vec p = vec(
            random_float(-1.0f ,1.0f , local_state),
            random_float(-1.0f ,1.0f , local_state), 
            0.0f
        );
        if(p.length_squared() >= 1) continue;

        return p ;
    }
}

class camera{
public:
    vec origin ;
    vec lower_left_corner;
    vec horizontal;
    vec vertical;
    vec u , v , w;  // u - camrera's right ; v - camera's up ; w - camera's backward
    
    float focus_dist;
    float lens_radius;

    __host__ __device__ camera(vec lookfrom , vec lookat , vec vup , float vfov ,float aspect_ratio , float aperture , float focus_dist) : focus_dist(focus_dist) {
        
        float theta = vfov * (3.14159265f / 180.0f);    // gpu requires radians 
        float h = tan(theta / 2.0f);    //  slicing the camera's view angle by half and warpping around tan func gives h
        float viewport_height = 2.0f * h;    // since h is only half of the total height so multiply by 2 to get full height 
        float viewport_width = aspect_ratio * viewport_height;  // this is the total width

        // camera's custom roatation axes
        w = unit_vector(lookfrom - lookat) ; 
        u = unit_vector(cross(vup , w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = viewport_width * focus_dist* u ;
        vertical = viewport_height * focus_dist * v ; 
        lower_left_corner = origin - (horizontal / 2.0f) - (vertical / 2.0f) -  (focus_dist * w);

        // aperture is diameter of the hole through which light enters 
        // for calculation we need radius
        lens_radius = aperture / 2.0f;  
    }

    __device__ ray get_ray(float s , float t , curandState* local_state)const {
        vec rd = lens_radius * random_unit_disk(local_state);
        vec offset = rd.x() * u + rd.y() * v ;

        return ray(origin + offset , lower_left_corner + s * horizontal + t* vertical - origin - offset);
    }
};
#endif