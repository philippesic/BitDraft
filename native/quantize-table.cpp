#include "ggml.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static ggml_type table_type(const std::string & name) {
    if (name == "F32") return GGML_TYPE_F32;
    if (name == "Q2_K") return GGML_TYPE_Q2_K;
    if (name == "Q3_K") return GGML_TYPE_Q3_K;
    if (name == "Q4_K") return GGML_TYPE_Q4_K;
    if (name == "Q5_K") return GGML_TYPE_Q5_K;
    if (name == "Q6_K") return GGML_TYPE_Q6_K;
    throw std::runtime_error("unsupported table type " + name);
}

static size_t row_bytes(ggml_type type, int64_t columns) {
    return type == GGML_TYPE_F32 ? size_t(columns) * sizeof(float) : ggml_row_size(type, columns);
}

static void read_rows(std::ifstream & input, std::vector<uint8_t> & buffer, size_t bytes) {
    buffer.resize(bytes);
    input.read(reinterpret_cast<char *>(buffer.data()), std::streamsize(bytes));
    if (!input) throw std::runtime_error("table input is shorter than declared");
}

static void decode(ggml_type type, const uint8_t * source, float * destination, int64_t rows, int64_t columns) {
    const size_t width = row_bytes(type, columns);
    if (type != GGML_TYPE_F32 && !ggml_validate_row_data(type, source, width * size_t(rows))) {
        throw std::runtime_error("source table rows are invalid");
    }
    for (int64_t row = 0; row < rows; ++row) {
        const uint8_t * line = source + width * size_t(row);
        float * out = destination + size_t(row) * size_t(columns);
        if (type == GGML_TYPE_F32) {
            std::copy(reinterpret_cast<const float *>(line), reinterpret_cast<const float *>(line) + columns, out);
        } else {
            ggml_get_type_traits(type)->to_float(line, out, columns);
        }
    }
}

int main(int argc, char ** argv) {
    try {
        if (argc != 7 && argc != 8) throw std::runtime_error("usage: binary-quantize-table SOURCE_TYPE TARGET_TYPE ROWS COLUMNS INPUT OUTPUT [ROWS_PER_CALL]");
        const ggml_type source_type = table_type(argv[1]);
        const ggml_type target_type = table_type(argv[2]);
        const int64_t rows = std::stoll(argv[3]);
        const int64_t columns = std::stoll(argv[4]);
        if (rows < 1 || columns < 1 || columns % 256 != 0) throw std::runtime_error("table geometry must be positive with 256-column blocks");
        if (target_type != GGML_TYPE_F32 && ggml_quantize_requires_imatrix(target_type)) throw std::runtime_error("target type needs an importance matrix");
        std::ifstream input(argv[5], std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("cannot open table input");
        const size_t source_width = row_bytes(source_type, columns);
        if (size_t(input.tellg()) != source_width * size_t(rows)) throw std::runtime_error("table input size differs from its declared geometry");
        input.seekg(0);
        std::ofstream output(argv[6], std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open table output");
        if (target_type != GGML_TYPE_F32) ggml_quantize_init(target_type);
        const int64_t chunk = 4096;
        const int64_t per_call = argc == 8 ? std::stoll(argv[7]) : chunk;
        if (per_call < 1 || per_call > chunk || chunk % per_call != 0) throw std::runtime_error("rows per call must divide 4096");
        const size_t target_width = row_bytes(target_type, columns);
        std::vector<uint8_t> encoded;
        std::vector<float> values;
        std::vector<uint8_t> coded;
        for (int64_t start = 0; start < rows; start += chunk) {
            const int64_t count = std::min(chunk, rows - start);
            read_rows(input, encoded, source_width * size_t(count));
            values.resize(size_t(count) * size_t(columns));
            decode(source_type, encoded.data(), values.data(), count, columns);
            if (target_type == GGML_TYPE_F32) {
                output.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
            } else {
                coded.resize(target_width * size_t(count));
                size_t written = 0;
                for (int64_t first = 0; first < count; first += per_call) {
                    const int64_t calls = std::min(per_call, count - first);
                    written += ggml_quantize_chunk(target_type, values.data() + size_t(first) * size_t(columns),
                                                   coded.data() + target_width * size_t(first), 0, calls, columns, nullptr);
                }
                if (written != coded.size() || !ggml_validate_row_data(target_type, coded.data(), coded.size())) {
                    throw std::runtime_error("quantized rows are invalid");
                }
                output.write(reinterpret_cast<const char *>(coded.data()), std::streamsize(coded.size()));
            }
            if (!output) throw std::runtime_error("cannot write table output");
        }
        output.close();
        if (!output) throw std::runtime_error("cannot close table output");
        std::cout << "{\"rows\":" << rows << ",\"columns\":" << columns << ",\"source\":\"" << argv[1]
                  << "\",\"target\":\"" << argv[2] << "\",\"rows_per_call\":" << per_call
                  << ",\"bytes\":" << target_width * size_t(rows) << "}" << std::endl;
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
