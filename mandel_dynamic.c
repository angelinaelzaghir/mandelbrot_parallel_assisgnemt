#include <mpi.h>
#include <stdio.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

#define WORK_TAG 1
#define STOP_TAG 2

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;

    int iter = 0;
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq = z_real2 + z_imag2;
        iter++;
    }
    while ((iter < MAX_ITER) && (lengthsq < 4.0));

    return iter;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE* pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n");
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);
    fprintf(pgmimg, "255\n");

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fprintf(pgmimg, "%d ", image[i][j]);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    if (rank == 0) {
        
        int image[HEIGHT][WIDTH];
        int next_row = 0;
        int active_workers = size - 1;

        
        for (int worker = 1; worker < size; worker++) {
            MPI_Send(&next_row, 1, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD);
            next_row++;
        }

        
        while (active_workers > 0) {
            int row_idx;
            int row_buffer[WIDTH];

            MPI_Status status;
            MPI_Recv(&row_idx, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            MPI_Recv(row_buffer, WIDTH, MPI_INT, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            
            for (int j = 0; j < WIDTH; j++) {
                image[row_idx][j] = row_buffer[j];
            }

            
            if (next_row < HEIGHT) {
                MPI_Send(&next_row, 1, MPI_INT, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);
                next_row++;
            } else {
                
                int stop_signal = -1;
                MPI_Send(&stop_signal, 1, MPI_INT, status.MPI_SOURCE, STOP_TAG, MPI_COMM_WORLD);
                active_workers--;
            }
        }

        save_pgm("mandelbrot_dynamic.pgm", image);

        double end_time = MPI_Wtime();
        printf("Dynamic scheduling runtime: %f seconds\n", end_time - start_time);
    }
    else {
        
        while (1) {
            int row_idx;
            MPI_Status status;
            MPI_Recv(&row_idx, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == STOP_TAG) {
                break;
            }

            int row_buffer[WIDTH];
            struct complex c;

            for (int j = 0; j < WIDTH; j++) {
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (row_idx - HEIGHT / 2.0) * 4.0 / HEIGHT;
                row_buffer[j] = cal_pixel(c);
            }

            MPI_Send(&row_idx, 1, MPI_INT, 0, WORK_TAG, MPI_COMM_WORLD);
            MPI_Send(row_buffer, WIDTH, MPI_INT, 0, WORK_TAG, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
