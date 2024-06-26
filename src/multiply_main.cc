#include <cstdio>

#include "common.h"
#include "multiply.h"
#include "encode.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

ABSL_FLAG(std::string, input_path, "", "Input file path");
ABSL_FLAG(std::string, input_vector_path, "", "Input vector path");
ABSL_FLAG(std::string, output_vector_path, "", "Output vector path");

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    // Ensure that encoder-only flags are recognized by the decoder too.
    (void) absl::GetFlag(FLAGS_allow_random_access);
    (void) absl::GetFlag(FLAGS_greedy_random_access);

    //data
    FILE *in = fopen(absl::GetFlag(FLAGS_input_path).c_str(), "r");
    ZKR_ASSERT(in);

    fseek(in, 0, SEEK_END);
    size_t len = ftell(in);
    fseek(in, 0, SEEK_SET);

    std::vector<uint8_t> data(len);
    ZKR_ASSERT(fread(data.data(), 1, len, in) == len);

    //invec
    FILE *in_invec = fopen(absl::GetFlag(FLAGS_input_vector_path).c_str(), "r");
    ZKR_ASSERT(in_invec);

    fseek(in_invec, 0, SEEK_END);
    size_t len_invec = ftell(in_invec);
    fseek(in_invec, 0, SEEK_SET);

    std::vector<double> invec(len_invec);
    ZKR_ASSERT(fread(invec.data(), 1, len_invec, in_invec) == len_invec);

    //outvec
    std::vector<double> outvec;
    if (!zuckerli::DecodeGraph(data, invec, outvec)) {
        fprintf(stderr, "Invalid graph\n");
        return EXIT_FAILURE;
    }

    //outfile
    FILE *out_outvec = fopen(absl::GetFlag(FLAGS_output_vector_path).c_str(), "wb");  // Open in binary format
    ZKR_ASSERT(out_outvec);
    fwrite(outvec.data(), sizeof(double), outvec.size(), out_outvec);
    fclose(out_outvec);  // Close the file

    return EXIT_SUCCESS;
}
