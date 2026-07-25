#ifndef VEC_H
#define VEC_H

#include<iostream>
#include<cmath>
class vec{
public:
    float e[3];
    __host__ __device__ vec() : e{0,0,0} {}
    __host__ __device__ vec(float e0 , float e1 ,float e2) : e{e0 , e1 , e2} {}

    __host__ __device__ float x() const {return e[0];}
    __host__ __device__ float y() const {return e[1];}
    __host__ __device__ float z() const {return e[2];}

    __host__ __device__ vec operator-() const{ return vec(-e[0], -e[1] , -e[2]); }
    __host__ __device__ float operator[](int i) const{return e[i] ;}
    __host__ __device__ float& operator[](int i) {return e[i] ;}

    __host__ __device__ vec& operator += (const vec& v){
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }
    __host__ __device__ vec& operator*= (const float t){
        e[0] *= t.e[0];
        e[1] *= t.e[1];
        e[2] *= t.e[2];
        return *this;
    }
    __host__ __device__ vec& operator/= (const float t){
        return *this *= (1/t) ;
    }
    __host__ __device__ float length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    __host__ __device__ float length() const {
        return sqrt(length_squared());
    }
};

inline std::ostream& operator<<(std::ostream& out , const vec& v){   // __host__ __device__ not used because it is cpu property
    return out << v.e[0] << ' ' << v.e[1] << ' '<< v.e[2] ;
}
__host__ __device__ inline vec& operator+(const vec& u , const vec& v){
    return vec(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}
__host__ __device__ inline vec& operator-(const vec& u , const vec& v){
    return vec(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}
__host__ __device__ inline vec& operator*(const vec& u , const vec& v){
    return vec(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}
__host__ __device__ inline vec& operator*(const vec& v , float t){
    return t * v;
}
__host__ __device__ inline vec& operator/(const vec& v , float t){
    return (1/t) * v;
}
__host__ __device__ inline float dot(const vec& u , const vec& v){
    return vec(u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2])
}
__host__ __device__ inline float cross(const vec& u , const vec& v){
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1] , u.e[2] * v.e[0] - u.e[0] * v.e[2] , u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}
__ host__ __device__ inline vec unit_vector(vec& v){
    return v / v.length()
}
#endif