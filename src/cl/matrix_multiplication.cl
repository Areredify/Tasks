#define TILE_SIZE 16

__kernel void matrix_multiplication(__global float * a, __global float * b, __global float * c, unsigned int M, unsigned int K, unsigned int N) {
    __local float local_a[TILE_SIZE][TILE_SIZE], local_b[TILE_SIZE][TILE_SIZE];

    unsigned int l_x = get_local_id(0), l_y = get_local_id(1);
    unsigned int g_x = get_global_id(0), g_y = get_global_id(1);

    float sum = 0;

    for (unsigned int k = 0; k * TILE_SIZE < K; k++) {
        unsigned int k_x = k * TILE_SIZE + l_x, k_y = k * TILE_SIZE + l_y;

        if (g_y < M && k_x < K)
            local_a[l_y][l_x] = a[g_y * K + k_x];
        else
            local_a[l_y][l_x] = 0;

        if (g_x < N && k_y < K)
            local_b[l_y][l_x] = b[k_y * N + g_x];
        else
            local_b[l_y][l_x] = 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int i = 0; i < TILE_SIZE; i++)
            sum += local_a[l_y][i] * local_b[i][l_x];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (g_y < M && g_x < N)
        c[g_y * N + g_x] = sum;
}