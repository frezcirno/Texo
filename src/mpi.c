#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // 初始化 MPI
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // 获取进程总数

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // 获取当前进程编号

    printf("Hello from rank %d out of %d processes\n", world_rank, world_size);

    MPI_Finalize(); // 结束 MPI
    return 0;
}
