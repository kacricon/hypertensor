#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include <assert.h>

/* Expected dimensions of names.txt */
#define NUM_SAMPLES     32033
#define MAX_NAME_LENGTH 15
#define NUM_CHARS       196113

#define DATASET_SIZE    NUM_CHARS + NUM_SAMPLES
#define TRAIN_SIZE      (DATASET_SIZE * 7 / 10)

/* Other definitions */
#define LIMIT_CHAR ('z' - 'a' + 1)

/* Load names.txt data from github to memory */
static bool load_data(const char* url, char names[NUM_SAMPLES][MAX_NAME_LENGTH + 2]) {
    char command[256];
    snprintf(command, sizeof(command), "curl -s %s", url);
    FILE* pipe = popen(command, "r");
    if (pipe == NULL) {
        fprintf(stderr, "Failed to open URL\n");
        return false;
    }

    int line_count = 0;
    while (line_count < NUM_SAMPLES && fgets(names[line_count] + 1, MAX_NAME_LENGTH + 2, pipe)) {
        size_t i;
        names[line_count][0] = LIMIT_CHAR;
        for (i = 1; names[line_count][i] != '\0' && names[line_count][i] != '\n'; i++) {
            names[line_count][i] -= 'a';
        }
        names[line_count][i] = LIMIT_CHAR;

        line_count++;
    }

    pclose(pipe);

    printf("Downloaded %d lines\n", line_count);
    return true;
}

/* converts array of names into input (x) and output (y) vectors */
static int prepare_dataset(char names[NUM_SAMPLES][MAX_NAME_LENGTH + 2], char x[DATASET_SIZE], char y[DATASET_SIZE]) {
    size_t dataset_size = 0;
    for (size_t i = 0; i < NUM_SAMPLES; i++) {
        if (dataset_size == DATASET_SIZE) {
            return dataset_size + 1; // signal early break for assert
        }

        size_t j;
        for (j = 0; names[i][j+1] != LIMIT_CHAR; j++) {
            x[dataset_size]   = names[i][j];
            y[dataset_size++] = names[i][j+1];
        }
        x[dataset_size]   = names[i][j];
        y[dataset_size++] = names[i][j+1];
    }
    return dataset_size;
}

int main() {
    // data preparation
    const char *url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt";
    char names[NUM_SAMPLES][MAX_NAME_LENGTH + 2];
    if (!load_data(url, names)) {
        return 1;
    }

    printf("\nFile contents:\n");
    for (int i = 0; i < 10; i++) {
        printf("Line %d: ", i+1);
        size_t limits = 0;
        for (size_t j = 0; limits < 2; j++) {
            printf("%d ", names[i][j]);
            if (names[i][j] == LIMIT_CHAR) limits++;
        }
        printf("\n");
    }

    char x[DATASET_SIZE], y[DATASET_SIZE];
    int dataset_size = prepare_dataset(names, x, y);
    assert(dataset_size == DATASET_SIZE);

    printf("\nDataset contents:\n");
    for (int i = 0; i < 20; i++) {
        printf("x[%d] = %d\ty[%d] = %d\n", i, x[i], i, y[i]);
    }

    return 0;
}

