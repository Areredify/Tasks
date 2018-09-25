// TODO

#define WORK_GROUP_SIZE 128

__kernel void max_prefix_sum(__global const int * sum_input, __global const int * prefix_input, const unsigned int input_size,
                             __global int * sum_output, __global int * prefix_output)
{
    unsigned int local_id = get_local_id(0),
                 global_id = get_global_id(0);

    __local int local_sum1[WORK_GROUP_SIZE];
    __local int local_prefix1[WORK_GROUP_SIZE];
    __local int local_sum2[WORK_GROUP_SIZE];
    __local int local_prefix2[WORK_GROUP_SIZE];

    __local int * local_sum = local_sum1;
    __local int * local_prefix = local_prefix1;
    __local int * other_sum = local_sum2;
    __local int * other_prefix = local_prefix2;
    __local int * tmp;
    
    if (input_size <= global_id) {
        local_sum[local_id] = 0;
        local_prefix[local_id] = 0;
    }
    else {
        local_sum[local_id] = sum_input[global_id];
        local_prefix[local_id] = prefix_input[global_id];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    int values_to_process = WORK_GROUP_SIZE;
    int power = 1;
    for (; values_to_process > 2; values_to_process /= 2)
    {
        if (local_id < values_to_process / 2) {
            other_prefix[local_id] = max(local_prefix[2 * local_id], local_sum[2 * local_id] + local_prefix[2 * local_id + 1]);
            other_sum[local_id] = local_sum[2 * local_id] + local_sum[2 * local_id + 1];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        tmp = other_prefix;
        other_prefix = local_prefix;
        local_prefix = tmp;

        tmp = other_sum;
        other_sum = local_sum;
        local_sum = tmp;
    }

    if (local_id == 0) {
        prefix_output[global_id / WORK_GROUP_SIZE] = max(local_prefix[0], local_sum[0] + local_prefix[1]);
        sum_output[global_id / WORK_GROUP_SIZE] = local_sum[0] + local_sum[1];
    }
}