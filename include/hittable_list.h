#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include"ray.h"
#include"hittable.h"
#include "random.h"

class hittable_list : public hittable{
public:
    hittable** list;
    int list_size;

    __host__ __device__ hittable_list() : list(nullptr) , list_size(0) {}
    __host__ __device__ hittable_list(hittable** l , int n) : list(l) , list_size(n) {}

    __host__ __device__ virtual bool hit(const ray& r , float t_min , float t_max , hit_record& rec , curandState* local_state) const override ;
    __host__ __device__ virtual bool bounding_box(aabb& output_box) const override;
};

__host__ __device__ bool hittable_list::hit(const ray& r , float t_min , float t_max , hit_record& rec , curandState* local_state) const{
    hit_record temp_rec ;
    bool hit_anything = false;

    float closest_so_far = t_max;

    for(int i = 0 ; i < list_size ; i++){
        if(list[i]->hit(r , t_min , closest_so_far , temp_rec , local_state)){
            hit_anything = true;
            closest_so_far = temp_rec.t;     // new closest distance
            rec = temp_rec;
        }
    }
    return hit_anything ;
}
__host__ __device__ bool hittable_list::bounding_box(aabb& output_box)const {
    if(list_size <= 0 || list == nullptr) return false;

    aabb temp_box;
    bool first_box = true;
    for(int i = 0 ; i < list_size ; i++){
        if(!list[i]->bounding_box(temp_box)){
            return false;
        }
        if(first_box){
            output_box = temp_box;
            first_box = false;
        }else{
            vec small(
                fminf(output_box.min().x() , temp_box.min().x()),
                fminf(output_box.min().y() , temp_box.min().y()),
                fminf(output_box.min().z() , temp_box.min().z())
            );
            vec large(
                fmaxf(output_box.max().x() , temp_box.max().x()),
                fmaxf(output_box.max().y() , temp_box.max().y()),
                fmaxf(output_box.max().z() , temp_box.max().z())
            );

            output_box = aabb(small , large);
        }
    }
    return true;
}
#endif