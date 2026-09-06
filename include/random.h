#ifndef RANDOM_H
#define RANDOM_H

#include <curand_kernel.h>
#include "vec.h"

__device__ inline float random_float(curandState* local_state) {
    return curand_uniform(local_state);
}

__device__ inline float random_float(
    float min,
    float max,
    curandState* local_state
) {
    return min + (max - min) * curand_uniform(local_state);
}
__device__ inline vec random_cosine_direction(curandState* local_state){
    float r1 = curand_uniform(local_state);   //  this is to pick a random number( btw 0 to 1)
    float r2 = curand_uniform(local_state);     // this is to pick a random dist from centre

    float phi = 2 * 3.14159265359f * r1;   // this converts it to random angle(0 to 360 deg)

    // we do sqrt(r2) as to spread out the points outwards countering the area changes(centre part has lesser area)
    float x = sqrtf(r2) * cosf(phi);   // in circle x is written as r * cos(theta)
    float y = sqrtf(r2) * sinf(phi);    // y is written as r * sin(theta)

    // malley's method : projecting upward
    float z = sqrtf(1- r2);
    return vec(x,y,z);
}
#endif