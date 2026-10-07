#ifndef RAY_H
#define RAY_H
#include "vec.h"
#include <cfloat>

class ray{
private:
    vec orig;
    vec dir;
    vec inv_dir;

public:
    __host__ __device__ ray()
        : orig(), dir(), inv_dir() {}

    // cache inverse dirn because AABB traversal performs this operation for a large number of nodes per ray
    __host__ __device__ ray(const vec& origin, const vec& direction)
        : orig(origin),
          dir(direction),
          inv_dir(
              fabsf(direction.x()) > 1e-8f ? 1.0f / direction.x() : FLT_MAX,
              fabsf(direction.y()) > 1e-8f ? 1.0f / direction.y() : FLT_MAX,
              fabsf(direction.z()) > 1e-8f ? 1.0f / direction.z() : FLT_MAX
          ) {}

    __host__ __device__ const vec& origin() const { return orig; }
    __host__ __device__ const vec& direction() const { return dir; }
    __host__ __device__ const vec& inv_direction() const { return inv_dir; }

    __host__ __device__ vec parametric_eqn(float t) const {
        return orig + t * dir;
    }
};
#endif