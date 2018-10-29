#define WORK_GROUP_SIZE 128
#define HALF_WG_SIZE (WORK_GROUP_SIZE / 2)

void swap_if_necessary( __global float *as, unsigned int i, unsigned int j, unsigned int gs ) {
    if (j < gs && as[i] > as[j]) {
        float t = as[j];
        as[j] = as[i];
        as[i] = t;
    }
}

void local_swap_if_necessary( __local float *as, unsigned int i, unsigned int j, unsigned int gs ) {
    if (j < gs && as[i] > as[j]) {
        float t = as[j];
        as[j] = as[i];
        as[i] = t;
    }
}

__kernel void bitonic(__global float* as, unsigned int n, unsigned int half_block_size, unsigned int first_pass) {
    const unsigned int global_id = get_global_id(0);

    const unsigned int block_start = global_id / half_block_size * half_block_size * 2, new_global_id = global_id + block_start / 2;

    if (first_pass == 1)
        swap_if_necessary(as, new_global_id, block_start + (-new_global_id + block_start + half_block_size * 2 - 1), n);
    else
        swap_if_necessary(as, new_global_id, new_global_id + half_block_size, n);
}

__kernel void bitonic_local(__global float* as, unsigned int n, unsigned int half_block_size, unsigned int first_pass) {
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    const unsigned int wg_block_start = global_id / WORK_GROUP_SIZE * WORK_GROUP_SIZE * 2;
    const unsigned int wg_local_id = global_id - wg_block_start / 2, wg_global_id = global_id + wg_block_start / 2;

    __local float local_mem[2 * WORK_GROUP_SIZE];

    if (wg_global_id < n)
        local_mem[wg_local_id] = as[wg_global_id];
    if (wg_global_id + WORK_GROUP_SIZE < n)
        local_mem[wg_local_id + WORK_GROUP_SIZE] = as[wg_global_id + WORK_GROUP_SIZE];

    barrier(CLK_LOCAL_MEM_FENCE);


    for (; half_block_size >= 1; half_block_size /= 2) {
        const unsigned int block_start = global_id / half_block_size * half_block_size * 2;
        const unsigned int block_global_id = global_id + block_start / 2;

        if (first_pass == 1) {
            first_pass = 0;
            local_swap_if_necessary(local_mem, block_global_id - wg_block_start, block_start + (-block_global_id + block_start + half_block_size * 2 - 1) - wg_block_start, n - wg_block_start);
        }
        else
            local_swap_if_necessary(local_mem, block_global_id - wg_block_start, block_global_id - wg_block_start + half_block_size, n - wg_block_start);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (wg_global_id < n)
        as[wg_global_id] = local_mem[wg_local_id];
    if (wg_global_id + WORK_GROUP_SIZE < n)
        as[wg_global_id + WORK_GROUP_SIZE] = local_mem[wg_local_id + WORK_GROUP_SIZE];
}

