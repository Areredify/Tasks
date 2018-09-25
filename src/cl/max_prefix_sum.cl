// TODO

#define WORK_GROUP_SIZE 128

__kernel void max_prefix_sum(__global const int * sum_input, __global const int * prefix_input, __global const int * ind_input, const unsigned int input_size,
                             __global int * sum_output, __global int * prefix_output, __global int * ind_output)
{
    unsigned int local_id = get_local_id(0),
                 global_id = get_global_id(0);

    __local int local_sum1[WORK_GROUP_SIZE];
    __local int local_prefix1[WORK_GROUP_SIZE];
    __local int local_ind1[WORK_GROUP_SIZE];
    __local int local_sum2[WORK_GROUP_SIZE];
    __local int local_prefix2[WORK_GROUP_SIZE];
    __local int local_ind2[WORK_GROUP_SIZE];

    __local int * local_sum = local_sum1;
    __local int * local_prefix = local_prefix1;
    __local int * local_ind = local_ind1;
    __local int * other_sum = local_sum2;
    __local int * other_prefix = local_prefix2;
    __local int * other_ind = local_ind2;
    __local int * tmp;
    
    if (input_size <= global_id) {
        local_sum[local_id] = 0;
        local_prefix[local_id] = 0;
        local_ind[local_id] = 0;
    }
    else {
        local_sum[local_id] = sum_input[global_id];
        local_prefix[local_id] = prefix_input[global_id]; 
        local_ind[local_id] = ind_input[global_id];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    int values_to_process = WORK_GROUP_SIZE;
    for (; values_to_process > 2; values_to_process /= 2) {
        if (local_id < values_to_process / 2) {
            int val_a = local_prefix[2 * local_id];
            int val_b = local_sum[2 * local_id] + local_prefix[2 * local_id + 1];

            other_prefix[local_id] = max(val_a, val_b);
            if (val_a >= val_b)
                other_ind[local_id] = local_ind[2 * local_id];
            else
                other_ind[local_id] = local_ind[2 * local_id + 1];
            other_sum[local_id] = local_sum[2 * local_id] + local_sum[2 * local_id + 1];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        tmp = other_prefix;
        other_prefix = local_prefix;
        local_prefix = tmp;

        tmp = other_sum;
        other_sum = local_sum;
        local_sum = tmp;

        tmp = other_ind;
        other_ind = local_ind;
        local_ind = tmp;
    }

    if (local_id == 0) {
        sum_output[global_id / WORK_GROUP_SIZE] = local_sum[0] + local_sum[1];
        prefix_output[global_id / WORK_GROUP_SIZE] = max(local_prefix[0], local_sum[0] + local_prefix[1]);

        if (local_prefix[0] >= local_sum[0] + local_prefix[1])
            ind_output[global_id / WORK_GROUP_SIZE] = local_ind[0];
        else
            ind_output[global_id / WORK_GROUP_SIZE] = local_ind[1];
    }
}


//// unoptimized version
//
//#define WORK_GROUP_SIZE 256
//
//__kernel void max_prefix_sum(__global const int * sum_input, __global const int * prefix_input, const unsigned int input_size,
//                             __global int * sum_output, __global int * prefix_output)
//{
//    unsigned int local_id = get_local_id(0),
//                 global_id = get_global_id(0);
//
//    __local int local_sum[WORK_GROUP_SIZE];
//    __local int local_prefix[WORK_GROUP_SIZE];
//    
//    if (input_size <= global_id) {
//        local_sum[local_id] = 0;
//        local_prefix[local_id] = 0;
//    }
//    else {
//        local_sum[local_id] = sum_input[global_id];
//        local_prefix[local_id] = prefix_input[global_id];
//    }
//
//    barrier(CLK_LOCAL_MEM_FENCE);
//    unsigned int values_to_process = WORK_GROUP_SIZE;
//    unsigned int power = 1;
//    for (; values_to_process > 2; values_to_process /= 2)
//    {
//        if (local_id < values_to_process / 2) {
//            local_prefix[2 * power * local_id] = max(local_prefix[2 * power * local_id], local_sum[2 * power * local_id] + local_prefix[2 * power * local_id + power]);
//            local_sum[2 * power * local_id] = local_sum[2 * power * local_id] + local_sum[2 * power * local_id + power];
//        }
//
//        barrier(CLK_LOCAL_MEM_FENCE);
//        power *= 2;
//    }
//
//    if (local_id == 0) {
//        prefix_output[global_id / WORK_GROUP_SIZE] = max(local_prefix[0], local_sum[0] + local_prefix[power]);
//        sum_output[global_id / WORK_GROUP_SIZE] = local_sum[0] + local_sum[power];
//    }
//}