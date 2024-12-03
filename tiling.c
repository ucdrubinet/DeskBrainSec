#include <vips/vips.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <time.h>

#define TILE_SIZE 1536

void save_and_tile(VipsImage *image_to_segment, const char *imagename, const char *output_dir, int tile_size) {
    char base_dir_name[256];
    snprintf(base_dir_name, sizeof(base_dir_name), "%s/%s", output_dir, imagename);
    printf("%s\n", base_dir_name);
    if (g_mkdir_with_parents(base_dir_name, 0755) != 0) {
        fprintf(stderr, "Failed to create directory: %s\n", base_dir_name);
        return;
    }

    if (vips_dzsave(image_to_segment, base_dir_name,
                    "layout", VIPS_FOREIGN_DZ_LAYOUT_GOOGLE,
                    "suffix", ".jpg[Q=90]",
                    "tile_size", tile_size,
                    "depth", VIPS_FOREIGN_DZ_DEPTH_ONE,
                    "properties", TRUE, NULL)) {
        vips_error_exit("Error saving tiles");
    }
}

int main(int argc, char *argv[]) {
    if (VIPS_INIT(argv[0]))
        vips_error_exit(NULL);

    const char *wsi_dir = argv[1];
    const char *save_dir = argv[2];

    DIR *dir = opendir(wsi_dir);
    if (!dir) {
        perror("Failed to open WSI directory");
        return EXIT_FAILURE;
    }

    struct dirent *entry;
    printf("Starting tiling....\n");

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type != DT_REG)
            continue;

        const char *imagename = entry->d_name;
        const char *extension = strrchr(imagename, '.');
        if (!extension)
            continue;

        char imagepath[256];
        snprintf(imagepath, sizeof(imagepath), "%s/%s", wsi_dir, imagename);
        
        clock_t start = clock();
        printf("Loading %s ......\n", imagename);
        VipsImage *vips_img = NULL;

        if (strcmp(extension, ".svs") == 0) {
            vips_img = vips_image_new_from_file(imagepath, "level", 0, NULL);
        } else {
            printf("Skipped, %s. This file is .svs, or not the file assigned\n", imagename);
            continue;
        }

        if (!vips_img) {
            fprintf(stderr, "Failed to load image: %s\n", imagepath);
            continue;
        }

        printf("Loaded Image: %s\n", imagepath);
        save_and_tile(vips_img, strtok(imagename, "."), save_dir, TILE_SIZE);
        g_object_unref(vips_img);

        printf("Done Tiling: %s\n", imagepath);
        printf("Processed in %.2f seconds\n", (double)(clock() - start) / CLOCKS_PER_SEC);
        printf("____________________________________________\n");
    }

    closedir(dir);
    vips_shutdown();
    return EXIT_SUCCESS;
}
