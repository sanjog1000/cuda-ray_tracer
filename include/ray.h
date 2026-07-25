#ifndef RAY_H
#define RAY_H
#include"vec.h"

class ray{
private:
    vec orig ;
    vec dir ;

public:
    // empty constructor 
    __host__ __device__ ray(){}
    
    //parameterized contructor
    __host__ __device__ ray(const vec& origin , const vec& direction) : orig(origin) , dir(direction) {}

    // getter function 
    __host__ __device__ vec origin() const {return orig;}
    __host__ __device__ vec direction() const{return dir;}

    // ray eqn : P(T) = A + t * B  
    __host__ __device__ vec parametric_eqn(float t) const {return (orig + dir * t) ;}
};
#endif