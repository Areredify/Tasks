// TODO
#define MAX_WORK_GROUP_SIZE 256

__kernel void sum(__global const unsigned int * input, __global unsigned int * res, unsigned int n) 
{
    unsigned int local_id = get_local_id(0),
                 global_id = get_global_id(0);

    __local unsigned int local_sum[MAX_WORK_GROUP_SIZE];
    if (n <= global_id)
        local_sum[local_id] = 0;
    else
        local_sum[local_id] = input[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned int values_to_add = get_local_size(0);
    for (; values_to_add > 8; values_to_add /= 2)
    {
        if (2 * local_id < values_to_add)
            local_sum[local_id] = local_sum[local_id] + local_sum[local_id + values_to_add / 2];

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0)
      atomic_add(res, local_sum[0] + local_sum[1] + local_sum[2] + local_sum[3] +
                      local_sum[4] + local_sum[5] + local_sum[6] + local_sum[7]);
}