#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define NUM_SAMPLES 32033
#define MAX_NAME_LENGTH 15

bool load_data(const char* url, char lines[NUM_SAMPLES][MAX_NAME_LENGTH + 1]) {
    char command[256];
    snprintf(command, sizeof(command), "curl -s %s", url);
    FILE* pipe = popen(command, "r");
    if (pipe == NULL) {
        fprintf(stderr, "Failed to open URL\n");
        return false;
    }

    int line_count = 0;
    while (line_count < NUM_SAMPLES && fgets(lines[line_count], MAX_NAME_LENGTH + 1, pipe)) {
        char *newline = strchr(lines[line_count], '\n'); // remove \n
        if (newline) {
            *newline = '\0';
        }
        line_count++;
    }

    pclose(pipe);

    printf("Downloaded %d lines\n", line_count);
    return true;
}

int main() {
    const char *url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt";
    char names[NUM_SAMPLES][MAX_NAME_LENGTH + 1];

    if (!load_data(url, names)) {
        return 1;
    }

    printf("File contents:\n");
    for (int i = 0; i < 100; i++) {
        printf("Line %d: %s\n", i+1, names[i]);
    }

    return 0;
}

