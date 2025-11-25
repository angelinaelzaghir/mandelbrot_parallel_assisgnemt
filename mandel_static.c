#include <mpi.h>
#include <stdio.h>
#include <time.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

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
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));
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

    int rank, size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = HEIGHT / size;
    int start = rank * rows_per_process;
    int end   = (rank == size - 1) ? HEIGHT : start + rows_per_process;

    int local_height = end - start;
    int local_image[local_height][WIDTH];

    struct complex c;

    double start_time = MPI_Wtime();

    for (int i = start; i < end; i++) {
        for (int j = 0; j < WIDTH; j++) {
            c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
            c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
            local_image[i - start][j] = cal_pixel(c);
        }
    }

    double end_time = MPI_Wtime();

    int full_image[HEIGHT][WIDTH];

    MPI_Gather(
        local_image, local_height * WIDTH, MPI_INT,
        full_image, local_height * WIDTH, MPI_INT,
        0, MPI_COMM_WORLD
    );

    if (rank == 0) {
        printf("Static MPI runtime: %f seconds\n", end_time - start_time);
        save_pgm("mandelbrot_static.pgm", full_image);
        printf("Image saved: mandelbrot_static.pgm\n");
    }

    MPI_Finalize();
    return 0;
}
