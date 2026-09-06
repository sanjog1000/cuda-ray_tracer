#ifndef AABB_H
#define AABB_H
#include "ray.h"
class aabb {
public:
    vec minimum;
    vec maximum;

    __device__ __host__ aabb(): minimum() , maximum() {}
    __device__ __host__ aabb(const vec& a , const vec& b) : minimum(a) , maximum(b) {}

    __device__ __host__ const vec& min() const {return minimum;}
    __device__ __host__ const vec& max() const {return maximum;}

    __device__ __host__ inline bool hit(const ray& r, float t_min ,float t_max) const{
        for(int a = 0 ; a< 3; a++){
            float direction = r.direction()[a];

            // ray is parallel to this pair of slabs (actually it is to prevent zero divide in 'direction')
            if(fabsf(direction) < 1e-8f){
                // case 1 : if the origin is outside the slab it can never enter the box
                if(r.origin()[a] < minimum[a] || r.origin()[a] > maximum[a]){
                    return false;
                }
                // case 2 : if origin is within the slabs then it may hit other axis
                //  so skip this whole iteration and move to other axis
                continue;
            }

            float invD = 1.0f / direction;
            float t0 = (min()[a] - r.origin()[a]) * invD ;
            float t1 = (max()[a] - r.origin()[a]) * invD ;
            
            if(invD < 0.0f){
                float temp = t0;
                t0 = t1;
                t1 = temp;
            }
            // t_min = max(t_min, t0);
            // t_max = min(t_max, t1);
            t_min = t_min > t0 ? t_min : t0;
            t_max = t_max > t1 ? t1 : t_max;

            if(t_max <= t_min) return false;
        }
        return true;
    }
};
#endif