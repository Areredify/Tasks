#define TILE_SIZE 16

__kernel void matrix_transpose(__global float * matr, __global float * matr_transpose, unsigned int M, unsigned int K) {
    __local float local_mem[TILE_SIZE * TILE_SIZE];

    unsigned int l_x = get_local_id(0), l_y = get_local_id(1);
    unsigned int g_x = get_global_id(0), g_y = get_global_id(1);

    if (g_x < K && g_y < M)
        local_mem[l_x * TILE_SIZE + l_y] = matr[g_y * K + g_x];

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int t_g_x = get_group_id(1) * TILE_SIZE + l_x,
                 t_g_y = get_group_id(0) * TILE_SIZE + l_y;

    if (t_g_x < M && t_g_y < K)
        matr_transpose[t_g_y * M + t_g_x] = local_mem[l_y * TILE_SIZE + l_x];
}