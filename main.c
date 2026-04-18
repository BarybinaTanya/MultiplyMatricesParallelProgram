#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define FILL_RANGE_MODULE_1 100
#define FILL_RANGE_MODULE_2 105

#define ROOT_PROCESS_NUMBER 0
#define NUMBER_SYSTEM_BASE_10 10
#define NET_DIMS_COUNT 2
#define SUCCESS 0
#define FAIL 1
#define TRUE 0

#define FILL_RANDOM_FLAG_1 0
#define FILL_RANDOM_FLAG_2 1
#define FILL_WITH_ZEROS_FLAG 2

#define NO_CYCLING 0
#define NO_REORDER 0

#define FIX_ROW 0
#define CHANGE_COL 1
#define FIX_COL 0
#define CHANGE_ROW 1

#define FIRST_COLUMN_COLOR 0
#define FIRST_ROW_COLOR 1

int global_ranks_count;
int ranks_net_x;
int ranks_net_y;
int N1_global;
int N2_global;
int N3_global;

void AllocateMatrix(double** matrix, int rows_number, int cols_number) {
    *matrix = (double*)malloc(rows_number * cols_number * sizeof(double));
}

void FillMatrix(double** matrix, int rows_number, int cols_number, int val_flag) {
    for (int i = 0; i < rows_number; ++i) {
        for (int j = 0; j < cols_number; ++j) {
            if (val_flag == 0) {
                (*matrix)[cols_number * i + j] = (double) (j % FILL_RANGE_MODULE_1);
            } else if (val_flag == 1) {
                (*matrix)[cols_number * i + j] = (double) (j % FILL_RANGE_MODULE_1) +
                                                 (double) (j % FILL_RANGE_MODULE_2);
            } else {
                (*matrix)[cols_number * i + j] = 0;
            }
        }
    }
}

int SequentialProgram() {
    double* matrix_a;
    AllocateMatrix(&matrix_a, N1_global, N2_global);
    double* matrix_b;
    AllocateMatrix(&matrix_b, N2_global, N3_global);
    double* matrix_r;
    AllocateMatrix(&matrix_r, N1_global, N3_global);
    if (matrix_a == NULL || matrix_b == NULL || matrix_r == NULL) {
        return FAIL;
    }
    FillMatrix(&matrix_a, N1_global, N2_global, FILL_RANDOM_FLAG_1);
    FillMatrix(&matrix_b, N2_global, N3_global, FILL_RANDOM_FLAG_2);
    FillMatrix(&matrix_r, N1_global, N3_global, FILL_WITH_ZEROS_FLAG);
    /*** for (int i = 0; i < N; i++)
             for (int j = 0; j < N; j++)
                 for (int k = 0; k < N; k++)
                        C[i][j] += A[i][k] * B[k][j];
     */
    double start, end;
    start = MPI_Wtime();
    for (int i = 0; i < N1_global; ++i) {
        for (int j = 0; j < N3_global; ++j) {
            for (int k = 0; k < N2_global; ++k) {
                matrix_r[i * N3_global + j] += matrix_a[i * N2_global + k] * matrix_b[k * N3_global + j];
            }
        }
    }
    end = MPI_Wtime();
    printf("Seq program finished, time: %lf\n", end - start);
    free(matrix_a);
    free(matrix_b);
    free(matrix_r);
    return SUCCESS;
}

int ParallelProgram() {
    int rank_world, size_world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
    MPI_Comm_size(MPI_COMM_WORLD, &size_world);

    if (size_world != ranks_net_x * ranks_net_y) {
        if (rank_world == ROOT_PROCESS_NUMBER)
            fprintf(stderr, "Error: number of processes must be %d x %d = %d\n",
                    ranks_net_x, ranks_net_y, ranks_net_x * ranks_net_y);
        MPI_Abort(MPI_COMM_WORLD, FAIL);
    }
    if (N1_global % ranks_net_x != 0 || N3_global % ranks_net_y != 0) {
        if (rank_world == ROOT_PROCESS_NUMBER)
            fprintf(stderr, "Error: N1 must be multiple "
                            "of ranks_net_x and N3 multiple of ranks_net_y\n");
        MPI_Abort(MPI_COMM_WORLD, FAIL);
    }

    // Creating cartesian topology MPI-communicator (MPI 2d-net), containing entire MPI_COMM_WORLD.
    int dims_sizes[NET_DIMS_COUNT] = {ranks_net_x, ranks_net_y};
    int periods[NET_DIMS_COUNT] = {NO_CYCLING, NO_CYCLING};
    MPI_Comm cart_topology_comm;
    MPI_Cart_create(MPI_COMM_WORLD, NET_DIMS_COUNT, dims_sizes,
                    periods, NO_REORDER, &cart_topology_comm);
    // NO_REORDER - MPI won't renumber processes to make them closer physically.

    int coords[NET_DIMS_COUNT];
    MPI_Cart_coords(cart_topology_comm, rank_world, NET_DIMS_COUNT, coords);
    int cur_proc_row = coords[0];
    int cur_proc_col = coords[1];

    // It will be also the size of the net's counted result blocks.
    int n1_local = N1_global / ranks_net_x;
    int n3_local = N3_global / ranks_net_y;
    double *local_A = (double*)malloc(n1_local * N2_global * sizeof(double));
    double *local_B = (double*)malloc(N2_global * n3_local * sizeof(double));
    double *local_C = (double*)calloc(n1_local * n3_local, sizeof(double));
    if (local_A == NULL || local_B == NULL || local_C == NULL) {
        return FAIL;
    }

    double *global_A = NULL, *global_B = NULL;
    if (cur_proc_row == 0 && cur_proc_col == 0) {
        global_A = (double*)malloc(N1_global * N2_global * sizeof(double));
        global_B = (double*)malloc(N2_global * N3_global * sizeof(double));
        FillMatrix(&global_A, N1_global, N2_global, FILL_RANDOM_FLAG_1);
        FillMatrix(&global_B, N2_global, N3_global, FILL_RANDOM_FLAG_2);
    }
    double program_core_start, program_core_end;
    program_core_start = MPI_Wtime();

    MPI_Comm zero_col_comm;
    MPI_Comm_split(cart_topology_comm, (cur_proc_col == 0) ? FIRST_COLUMN_COLOR : MPI_UNDEFINED,
                   cur_proc_row, &zero_col_comm);
    // The key gives the split-procedure possibility to rang processes inside new coming communicator.
    if (cur_proc_col == 0) {
        // zero column MPI_Scatter:
        MPI_Scatter(global_A, n1_local * N2_global, MPI_DOUBLE,
                    local_A, n1_local * N2_global, MPI_DOUBLE,
                    0, zero_col_comm);
    }
    if (zero_col_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&zero_col_comm);
    }

    // Setting general geometry of the new MPI-datatype. MPI will know, how to find elements of
    // current slice of process n inside the matrix.
    MPI_Datatype slice_column_general_geometry_type;
    int blocks_count = N2_global;

    MPI_Type_vector(blocks_count, n3_local, N3_global,
                    MPI_DOUBLE, &slice_column_general_geometry_type);
    MPI_Type_commit(&slice_column_general_geometry_type);
    // Now MPI "thinks" that i + 1 slice is located at end of i column, but it's wrong.
    // Slice for i + 1 process is located at global_B[0][0] + (number columns at slice = local_n3) * sizeof(double)
    MPI_Datatype complete_slice_column_type;
    MPI_Type_create_resized(slice_column_general_geometry_type, 0,
                            n3_local * sizeof(double),
                            &complete_slice_column_type);
    MPI_Type_commit(&complete_slice_column_type);
    MPI_Type_free(&slice_column_general_geometry_type);

    MPI_Comm zero_row_comm;
    MPI_Comm_split(cart_topology_comm, (cur_proc_row == 0) ? FIRST_ROW_COLOR : MPI_UNDEFINED,
                   cur_proc_col, &zero_row_comm);

    if (cur_proc_row == 0) {
        if (cur_proc_col == 0) {
            for (int j = 1; j < ranks_net_y; j++) {
                MPI_Send(global_B, 1, complete_slice_column_type, j, 0, cart_topology_comm);
            }
            for (int i = 0; i < N2_global; i++) {
                for (int j = 0; j < n3_local; j++) {
                    local_B[i * n3_local + j] = global_B[i * N3_global + j];
                }
            }
        } else {
            MPI_Recv(local_B, N2_global * n3_local, MPI_DOUBLE, 0, 0,
                     cart_topology_comm, MPI_STATUS_IGNORE);
        }
    }
    MPI_Type_free(&complete_slice_column_type);

    if (cur_proc_row == 0 && cur_proc_col == 0) {
        free(global_A);
        free(global_B);
    }

    MPI_Comm row_communicator, col_communicator;
    int remain_dims[NET_DIMS_COUNT] = {FIX_ROW, CHANGE_COL};
    // MPI_Cart_sub cuts the process-net to subnet with fewer dimensions.
    MPI_Cart_sub(cart_topology_comm, remain_dims, &row_communicator);
    int remain_dims_col[NET_DIMS_COUNT] = {CHANGE_ROW, FIX_COL};
    MPI_Cart_sub(cart_topology_comm, remain_dims_col, &col_communicator);

    MPI_Bcast(local_A, n1_local * N2_global, MPI_DOUBLE, 0, row_communicator);
    MPI_Bcast(local_B, N2_global * n3_local, MPI_DOUBLE, 0, col_communicator);
    MPI_Comm_free(&row_communicator);
    MPI_Comm_free(&col_communicator);

    for (int i = 0; i < n1_local; i++) {
        for (int j = 0; j < n3_local; j++) {
            double sum = 0.0;
            for (int k = 0; k < N2_global; k++) {
                sum += local_A[i * N2_global + k] * local_B[k * n3_local + j];
            }
            local_C[i * n3_local + j] = sum;
        }
    }

    double *global_C = NULL;
    if (cur_proc_row == 0 && cur_proc_col == 0) {
        global_C = (double*)malloc(N1_global * N3_global * sizeof(double));
    }

    int global_C_size[2] = {N1_global, N3_global};
    int sub_matrix_size[2] = {n1_local, n3_local};

    if (cur_proc_row == 0 && cur_proc_col == 0) {
        if (global_C == NULL) {
            return FAIL;
        }
        // Copy our own counted data.
        for (int i = 0; i < n1_local; i++) {
            for (int j = 0; j < n3_local; j++) {
                global_C[i * N3_global + j] = local_C[i * n3_local + j];
            }
        }

        for (int r = 0; r < ranks_net_x; r++) {
            for (int c = 0; c < ranks_net_y; c++) {
                if (r == 0 && c == 0) {
                    continue;
                }
                int target_coords[2] = {r, c};
                int target_rank;
                MPI_Cart_rank(cart_topology_comm, target_coords, &target_rank);

                MPI_Datatype temp_subarray;
                int current_starts[2] = {r * n1_local, c * n3_local};
                MPI_Type_create_subarray(2, global_C_size, sub_matrix_size, current_starts,
                                         MPI_ORDER_C, MPI_DOUBLE, &temp_subarray);
                MPI_Type_commit(&temp_subarray);

                MPI_Recv(global_C, 1, temp_subarray, target_rank,
                         0, cart_topology_comm, MPI_STATUS_IGNORE);
                MPI_Type_free(&temp_subarray);
            }
        }
    } else {
        int starts[2] = {cur_proc_row * n1_local, cur_proc_col * n3_local};
        MPI_Datatype subarray_C;
        MPI_Type_create_subarray(2, global_C_size, sub_matrix_size, starts,
                                 MPI_ORDER_C, MPI_DOUBLE, &subarray_C);
        MPI_Type_commit(&subarray_C);

        int root_rank;
        int root_coords[2] = {0, 0};
        MPI_Cart_rank(cart_topology_comm, root_coords, &root_rank);
        MPI_Send(local_C, 1, subarray_C, root_rank, 0, cart_topology_comm);
        MPI_Type_free(&subarray_C);
    }
    if (cur_proc_row == 0 && cur_proc_col == 0) {
        free(global_C);
    }
    program_core_end = MPI_Wtime();
    if (cur_proc_col == 0 && cur_proc_row == 0) {
        printf("Program finished, Time: %lf\n", program_core_end - program_core_start);
    }
    free(local_A);
    free(local_B);
    free(local_C);
    MPI_Comm_free(&cart_topology_comm);

    return SUCCESS;
}

int ReadNumeralArguments(char *argv[], int arg_num, int rank, int* variable_ptr) {
    char *end;
    if (rank == ROOT_PROCESS_NUMBER) {
        long val = strtol(argv[arg_num], &end, NUMBER_SYSTEM_BASE_10);
        if (end == argv[arg_num] || *end != '\0') {
            printf("Invalid arguments0\n");
            return FAIL;
        }
        if (variable_ptr == NULL) {
            printf("Invalid arguments1\n");
            return FAIL;
        }
        *variable_ptr = (int) val;
    }
    MPI_Bcast(variable_ptr, 1, MPI_INT, ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);
    return SUCCESS;
}

int main(int argc, char *argv[]) {
    int ret = SUCCESS;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &global_ranks_count);

    if (argc != 7) {
        printf("Invalid arguments2\n");
        return FAIL;
    }

    int arg_num = 1;
//    if (ReadNumeralArguments(argc, argv, arg_num, rank, &global_ranks_count) == 1) {
//        printf("Invalid arguments3\n");
//        return FAIL;
//    }

    if (ReadNumeralArguments(argv, arg_num, rank, &ranks_net_x) == 1) {
        printf("Invalid arguments4\n");
        return FAIL;
    }
    arg_num = 2;
    if (ReadNumeralArguments(argv, arg_num, rank, &ranks_net_y) == 1) {
        printf("Invalid arguments5\n");
        return FAIL;
    }
    arg_num = 3;
    if (ReadNumeralArguments(argv, arg_num, rank, &N1_global) == 1) {
        printf("Invalid arguments6\n");
        return FAIL;
    }
    arg_num = 4;
    if (ReadNumeralArguments(argv, arg_num, rank, &N2_global) == 1) {
        printf("Invalid arguments7\n");
        return FAIL;
    }
    arg_num = 5;
    if (ReadNumeralArguments(argv, arg_num, rank, &N3_global) == 1) {
        printf("Invalid arguments8\n");
        return FAIL;
    }

    arg_num = 6;
    if (strcmp(argv[arg_num], "-s") == TRUE) {
        if (rank == ROOT_PROCESS_NUMBER) {
            ret = SequentialProgram();
        }
    } else if (strcmp(argv[arg_num], "-p") == TRUE) {
        ret = ParallelProgram();
    } else {
        if (rank == ROOT_PROCESS_NUMBER) {
            printf("Invalid arguments9\n");
            ret = FAIL;
        }
    }
//    printf("%d %d %d %d %d %d %s\n", global_ranks_count, ranks_net_x, ranks_net_y,
//           N1_global, N2_global, N3_global, argv[arg_num]);
    MPI_Finalize();
    return ret;
}
