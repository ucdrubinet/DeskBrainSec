CFLAGS = $(shell pkg-config --cflags vips)
LIBS = $(shell pkg-config --libs vips)
# NOTE: replace with path to your folder with .dylib files
# If you installed the python libvips package with conda-forge, then the Vips
# dylibs are in your anaconda3 folder
# Ex: /usr/anaconda3/envs/brainsec/lib
RPATH = /user/anaconda3/envs/brainsec/lib

tiling: tiling.c
	gcc $(CFLAGS) tiling.c -o tiling_program $(LIBS) -Wl,-rpath,$(RPATH)