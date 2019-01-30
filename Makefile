# _*_MakeFile_*_
CC = gcc
_CC2 = nvcc
CFLAGS = -Wall
_CC2FLAGS = -arch=sm_50 
# tiger gpu is sm_60, Molesky laptop sm_50

_MAJORV = 1
_MINORV = 0.1

_COPTS = -std=c99

IDIR = -I. -I/usr/include
_LIBDIR = -L. -L/usr/lib/x86_64-linux-gnu
_LIBS_C = -lm -l$(_G_SHARED_LIB)
_LIBS_CUDA = -lm -lcufft -lcublas -lcusolver

gDMR: testLS.o gDMR.o
	$(_CC2) $(_CC2FLAGS) -shared -Xlinker '-soname,libgDMR.so.$(_MAJORV)' -o libgDMR.so.$(_MAJORV).$(_MINORV) testLS.o gDMR.o $(_LIBDIR) $(_LIBS_CUDA)
	ln -sf libgDMR.so.$(_MAJORV).$(_MINORV) libgDMR.so.$(_MAJORV)
	ln -sf libgDMR.so.$(_MAJORV) libgDMR.so 

testLS.o:
	$(_CC2) $(_CC2FLAGS) -Xcompiler '-fPIC' -dc testLS.cu $(IDIR)

gDMR.o:
	$(_CC2) $(_CC2FLAGS) -Xcompiler '-fPIC' -dc gDMR.cu $(IDIR)

progDMR:
	$(CC) $(_COPTS) -o progDMR.exe progDMR.c $(IDIR) -Wl,-rpath,'$$ORIGIN' $(_LIBDIR) -lgDMR -lm

.PHONY: cleanLibs

.PHONY: cleanObjs

.PHONY: cleanProgs

cleanLibs: 
	rm -f ./*.so ./*.so.* ./*.o

cleanProgs:
	rm -f ./*.exe ./*.o

cleanObjs:
	mv ./*.o ./bin