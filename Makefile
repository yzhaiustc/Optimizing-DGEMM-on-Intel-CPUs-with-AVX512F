BINARY_NAME = dgemm_x86
CC			= icc
CFLAGS		= -O3 -march=native
MKLPATH		= /opt/intel/mkl
LDFLAGS		= -L$(MKLPATH)/lib/intel64 -mkl=sequential -lpthread -lm -ldl -DMKL_ILP64
INCFLAGS	= -I$(MKLPATH)/include


SRC			= $(wildcard *.c)
build : $(BINARY_NAME)

$(BINARY_NAME): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCFLAGS) $(SRC) -o $(BINARY_NAME)

clean:
	rm $(BINARY_NAME)