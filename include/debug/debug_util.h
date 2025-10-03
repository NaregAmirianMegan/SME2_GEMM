#include <iostream>
#include <iomanip>
#include <arm_sme.h>

template<typename DTYPE>
void display_matrix(const DTYPE* mat, int m, int n) {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			std::cout << std::setw(3) << mat[n*i+j] << " ";
		}
	std::cout << std::endl;
	}
}

// DEBUG ZA store utility
template<typename DTYPE, int C, int ELEMS_PER_VEC, int TILE_ROW, int VEC_IN_TILE, int TILE_COL>
static void inline dbg_store_za_vec_helper(DTYPE* row_ptr) __arm_streaming __arm_inout("za") {
    constexpr uint32_t TILE_ID = TILE_ROW * C + TILE_COL;
    DTYPE* col_ptr = row_ptr + TILE_COL * ELEMS_PER_VEC;
    svbool_t all_true = svptrue_b32();
    svst1_hor_za32(TILE_ID, VEC_IN_TILE, all_true, col_ptr);
}

template<typename DTYPE, int R, int C, int ELEMS_PER_VEC, int TILE_ROW, int VEC_IN_TILE, int... TileCols>
static void inline dbg_store_za_vec(DTYPE* buf, 
                                    std::integer_sequence<int, TileCols...>)
    __arm_streaming __arm_inout("za") {
    
    constexpr int row_idx = TILE_ROW * ELEMS_PER_VEC + VEC_IN_TILE;

    DTYPE* row_ptr = buf + row_idx * (R*ELEMS_PER_VEC);

    ((dbg_store_za_vec_helper<DTYPE, C, ELEMS_PER_VEC, TILE_ROW, VEC_IN_TILE, TileCols>(row_ptr)), ...);
}

template<typename DTYPE, int R, int C, int ELEMS_PER_VEC, int TILE_ROW, int... VecIndices, int... TileCols>
static void inline dbg_store_za_row(DTYPE* buf,
                                    std::integer_sequence<int, VecIndices...>,
                                    std::integer_sequence<int, TileCols...> cols)
    __arm_streaming __arm_inout("za") {
    
    ((dbg_store_za_vec<DTYPE, R, C, ELEMS_PER_VEC, TILE_ROW, VecIndices>(
        buf, cols)), ...);
}

template<typename DTYPE, int R, int C, int ELEMS_PER_VEC, int... TileRows, int... VecIndices, int... TileCols>
static void inline dbg_store_za_impl(DTYPE* buf,
                                     std::integer_sequence<int, TileRows...>,
                                     std::integer_sequence<int, VecIndices...> vecs,
                                     std::integer_sequence<int, TileCols...> cols)
    __arm_streaming __arm_inout("za") {
    
    ((dbg_store_za_row<DTYPE, R, C, ELEMS_PER_VEC, TileRows>(
        buf, vecs, cols)), ...);
}

template<typename DTYPE, int R, int C, int ELEMS_PER_VEC>
static void inline dbg_store_za(DTYPE* buf)
    __arm_streaming __arm_inout("za") {
    
    dbg_store_za_impl<DTYPE, R, C, ELEMS_PER_VEC>(
 		buf,
        std::make_integer_sequence<int, R>{},
        std::make_integer_sequence<int, ELEMS_PER_VEC>{},
        std::make_integer_sequence<int, C>{}
    );
}

// ISSUE: add output stream as argument to stream data to (like a log file)
template<typename DTYPE, int SVL, int R, int C>
void dump_za() __arm_streaming __arm_inout("za") {
	constexpr int ELEMS_PER_VEC = SVL / sizeof(DTYPE);
	constexpr int BUFSIZE = R * C * ELEMS_PER_VEC * ELEMS_PER_VEC * sizeof(DTYPE);

	static DTYPE buf[BUFSIZE];

	// read za into that buffer
	dbg_store_za<DTYPE, R, C, ELEMS_PER_VEC>((DTYPE*)buf);

	// display formatted ZA data from buffer to output stream (in the future we can pass this in as a parameter)
	for (int i = 0; i < R*ELEMS_PER_VEC; ++i) {
		for (int j = 0; j < C*ELEMS_PER_VEC; ++j) {
			std::cout << std::setw(3) << buf[i*(R*ELEMS_PER_VEC)+j] << " ";
		}
		std::cout << std::endl;
	}
}
