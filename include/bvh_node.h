#ifndef BVH_NODE_H
#define BVH_NODE_H

#include "hittable.h"
#include "aabb.h"
#include <curand_kernel.h>

// randomly choose one of the three spatial axes
// 0 -> X, 1 -> Y, 2 -> Z
__device__ inline int random_axis(curandState* local_state){
    return curand(local_state) % 3;
}

// Compare two objects using the minimum coordinate of their bounding boxes along the selected axis
__device__ inline bool compare(hittable* a , hittable* b , int axis){
    // returns true if a comes before b
    aabb box_a , box_b;

    a->bounding_box(box_a);
    b->bounding_box(box_b);

    if(axis == 0){
        return box_a.min().x() < box_b.min().x();
    }else if(axis == 1){
        return box_a.min().y() < box_b.min().y();
    }else{
        return box_a.min().z() < box_b.min().z();
    }
}
// sort objects in the range [start, end)
__device__ inline void sort_objects(hittable** src_objects , int start , int end , int axis){
    int left = start ; 
    int right = end -1;

    if(left >= right) return;

    hittable* pivot = src_objects[left + (right - left)/2];

    while(left <= right){
        while(compare(src_objects[left] , pivot , axis)){
            left++;
        }
        while(compare(pivot , src_objects[right] , axis)){
            right--;
        }

        if(left <= right){
            // swap
            hittable* temp = src_objects[left];
            src_objects[left] = src_objects[right];
            src_objects[right] = temp;
            left++;
            right--;
        }
    }

    sort_objects(src_objects , start , right+1 , axis);
    sort_objects(src_objects , left , end , axis);
}

#if defined(__CUDACC__)
__device__ inline void* operator new(size_t , void* ptr){return ptr;}
#endif

class bvh_node : public hittable {
public:
    hittable* left;
    hittable* right;
    aabb box;

    __host__ __device__ bvh_node() : left(nullptr) , right (nullptr) , box() {}
    __host__ __device__ bvh_node(hittable* l , hittable* r ,aabb b) : left(l) , right(r) , box(b) {}

    __host__ __device__ virtual bool hit(const ray& r , float t_min , float t_max , hit_record& rec , curandState* local_state) const override{
        // Return immediately if it failed to hit the giant box 
        if(!box.hit(r , t_min , t_max)){
            return false;
        }
        // leaf node stor the same obj in left and right 
        if(left == right){
            return left->hit(r , t_min , t_max , rec , local_state);
        }

        // if the big box was : check the left and right sub-tree 
        bool hit_left = left->hit(r , t_min , t_max , rec , local_state);

        // if the left box is hit --> shrink the t_max as anything beyond wont be visible n useless to calculate 
        // but if not hit then keep it as t_max as it may hit the right box 
        float closest_allowed = hit_left ? rec.t : t_max ; 
        // check the right box hit . 
        // here the use closest_allowed depending on whether left box was hit or not
        bool hit_right = right->hit(r , t_min , closest_allowed , rec , local_state);

        return hit_left || hit_right ;
    }
    __host__ __device__ static aabb surrounding_box(const aabb& box0 , const aabb& box1){
        vec small(fminf(box0.min().x() , box1.min().x()),
                fminf(box0.min().y() , box1.min().y()),
                fminf(box0.min().z() , box1.min().z()));

        vec large(fmaxf(box0.max().x() , box1.max().x()),
                fmaxf(box0.max().y() , box1.max().y()),
                fmaxf(box0.max().z() , box1.max().z()));        

        return aabb(small ,large);

    }
    __host__ __device__ virtual bool bounding_box(aabb& output_box) const override {
        output_box = box;
        return true;
    }
};
    __device__ bvh_node* build_bvh(hittable** src_objects , int start , int end , curandState* local_state , bvh_node* node_pool , int* pool_index){
        int obj_span = end - start;
        int axis = random_axis(local_state);

        sort_objects(src_objects , start ,end , axis);
        
        // dynamic allocation(new) is very slow as it locks single global heap memory 
        // so pre allocated memory pool
        // also memory leak problem using standard 'new' -- clear_world() --> clears the objecg nodes but not the internal bvh_nodes created 
        bvh_node* node = new(&node_pool[*pool_index]) bvh_node();
        (*pool_index)++ ;

        if(obj_span == 1){
            node->left = node->right = src_objects[start];
        }
        else if(obj_span == 2){
            node->left = src_objects[start];
            node->right = src_objects[start+1];
        }
        else{
            int mid = start + (obj_span / 2);

            node->left = build_bvh(src_objects , start , mid , local_state , node_pool , pool_index);
            node->right = build_bvh(src_objects , mid , end , local_state , node_pool , pool_index);
        }

        aabb box_left , box_right;
        node->left->bounding_box(box_left);
        node->right->bounding_box(box_right);

        node->box = node->surrounding_box(box_left,box_right);

        return node;
    }

#endif